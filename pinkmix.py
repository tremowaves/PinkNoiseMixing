#!/usr/bin/env python3
"""
Pink Noise Audio Normalizer
Ứng dụng cân bằng âm thanh với Pink Noise Reference
Hỗ trợ nhiều reference level cho các loại audio khác nhau

GUI Version
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import json
from pathlib import Path

# GUI specific imports
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
# Import required for robust drag-and-drop functionality
try:
    from tkinterdnd2 import TkinterDnD
except ImportError:
    print("Error: tkinterdnd2 library not found. Please install it using: pip install tkinterdnd2-py")
    sys.exit(1)


# --- Core Logic ---

# Thiết lập logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pink_noise_normalizer.log', mode='w'), # Overwrite log each run
    ]
)
logger = logging.getLogger(__name__)

class PinkNoiseGenerator:
    """
    Generator tạo Pink Noise với các thuật toán khác nhau
    """
    @staticmethod
    def generate_frequency_domain(n_samples, sample_rate):
        """Tạo pink noise trong miền tần số"""
        white_noise = np.random.randn(n_samples)
        white_fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples, d=1/sample_rate)
        
        scale = np.ones_like(freqs)
        if freqs[0] == 0:
            scale[1:] = 1 / np.sqrt(freqs[1:])
        else:
            scale = 1 / np.sqrt(freqs)

        pink_fft = white_fft * scale
        pink_noise = np.fft.irfft(pink_fft, n=n_samples)
        return pink_noise

    @staticmethod
    def generate_with_target_lufs(duration_seconds, sample_rate, target_lufs):
        """Tạo pink noise với target LUFS level cụ thể"""
        n_samples = int(duration_seconds * sample_rate)
        pink_noise = PinkNoiseGenerator.generate_frequency_domain(n_samples, sample_rate)
        
        meter = pyln.Meter(sample_rate)
        current_lufs = meter.integrated_loudness(pink_noise)
        
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        pink_noise_normalized = pink_noise * gain_linear
        
        if np.max(np.abs(pink_noise_normalized)) > 0.98:
            safety_factor = 0.98 / np.max(np.abs(pink_noise_normalized))
            pink_noise_normalized *= safety_factor
            logger.warning(f"Applied safety limiting for target LUFS {target_lufs}")
        
        return pink_noise_normalized

class AudioProcessor:
    """
    Xử lý audio matching với pink noise reference
    """
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg']
    
    def find_audio_files(self, folder_path):
        """Tìm tất cả file audio trong thư mục và thư mục con"""
        audio_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_formats):
                    full_path = os.path.join(root, file)
                    audio_files.append(full_path)
        logger.info(f"Tìm thấy {len(audio_files)} file audio")
        return audio_files
    
    def analyze_frequency_spectrum(self, audio, sr, n_fft=2048):
        """Phân tích phổ tần số của audio"""
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//4)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        avg_spectrum = np.mean(magnitude_db, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return freqs, avg_spectrum
    
    def calculate_gain_adjustments(self, audio_spectrum, pink_spectrum):
        """Tính toán gain adjustments cần thiết. Chỉ attenuate, không boost."""
        gain_diff = audio_spectrum - pink_spectrum
        gain_adjustments = np.where(gain_diff > 0, -gain_diff, 0)
        gain_adjustments = gaussian_filter1d(gain_adjustments, sigma=2)
        return gain_adjustments
    
    def apply_spectral_adjustment(self, audio, sr, gain_adjustments, n_fft=2048):
        """Áp dụng spectral adjustment"""
        hop_length = n_fft // 4
        audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')
        stft = librosa.stft(audio_padded, n_fft=n_fft, hop_length=hop_length)
        
        gain_linear = 10 ** (gain_adjustments / 20)
        gain_linear = gain_linear.reshape(-1, 1)
        
        stft_adjusted = stft * gain_linear
        audio_adjusted = librosa.istft(stft_adjusted, hop_length=hop_length, length=len(audio_padded))
        
        audio_adjusted = audio_adjusted[n_fft // 2 : len(audio) + n_fft // 2]
        return audio_adjusted
    
    def match_loudness_to_target(self, audio, sr, target_lufs):
        """Match loudness của audio với target LUFS"""
        meter = pyln.Meter(sr)
        try:
            current_lufs = meter.integrated_loudness(audio)
            if np.isinf(current_lufs) or np.isnan(current_lufs):
                logger.warning("Audio is too silent to measure LUFS, skipping loudness matching.")
                return audio

            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            audio_normalized = audio * gain_linear
            
            if np.max(np.abs(audio_normalized)) > 0.98:
                peak_gain = 0.98 / np.max(np.abs(audio_normalized))
                audio_normalized *= peak_gain
                logger.warning(f"Applied peak limiting: {20*np.log10(peak_gain):.2f} dB")
            
            return audio_normalized
        except Exception as e:
            logger.error(f"Error matching loudness: {e}")
            return audio
    
    def _process_channel(self, audio_channel, sr, pink_reference_lufs, use_spectral_matching):
        """Xử lý một channel audio"""
        if use_spectral_matching:
            duration = len(audio_channel) / sr
            pink_ref = PinkNoiseGenerator.generate_with_target_lufs(duration, sr, pink_reference_lufs)
            _, audio_spectrum = self.analyze_frequency_spectrum(audio_channel, sr)
            _, pink_spectrum = self.analyze_frequency_spectrum(pink_ref, sr)
            gain_adjustments = self.calculate_gain_adjustments(audio_spectrum, pink_spectrum)
            audio_adjusted = self.apply_spectral_adjustment(audio_channel, sr, gain_adjustments)
        else:
            audio_adjusted = audio_channel
        
        audio_final = self.match_loudness_to_target(audio_adjusted, sr, pink_reference_lufs)
        return audio_final

    def process_single_file(self, audio_file, pink_reference_lufs, output_folder, use_spectral_matching=True):
        """Xử lý một file audio"""
        try:
            data, sr = librosa.load(audio_file, sr=None, mono=False)
            
            if data.ndim == 2:
                processed_channels = [
                    self._process_channel(data[ch], sr, pink_reference_lufs, use_spectral_matching) 
                    for ch in range(data.shape[0])
                ]
                processed_audio = np.array(processed_channels)
            else:
                processed_audio = self._process_channel(data, sr, pink_reference_lufs, use_spectral_matching)
            
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(output_folder, f"normalized_{base_name}.wav")
            
            sf.write(output_path, processed_audio.T if processed_audio.ndim == 2 else processed_audio, sr)
            return True
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            return False

class PresetManager:
    """
    Quản lý presets cho các loại audio khác nhau
    """
    DEFAULT_PRESETS = {
        'sfx': {'name': 'Sound Effects', 'target_lufs': -12.0, 'description': 'Tối ưu cho hiệu ứng âm thanh và foley', 'use_spectral_matching': True},
        'music': {'name': 'Music', 'target_lufs': -8.0, 'description': 'Tối ưu cho nhạc', 'use_spectral_matching': True},
        'voice': {'name': 'Voice/Dialog', 'target_lufs': -16.0, 'description': 'Tối ưu cho giọng nói và hội thoại', 'use_spectral_matching': False},
        'broadcast': {'name': 'Broadcast', 'target_lufs': -23.0, 'description': 'Chuẩn broadcast EBU R128', 'use_spectral_matching': True},
        'streaming': {'name': 'Streaming', 'target_lufs': -14.0, 'description': 'Tối ưu cho các nền tảng streaming', 'use_spectral_matching': True}
    }
    
    def __init__(self, config_file='presets.json'):
        self.config_file = config_file
        self.presets = self.load_presets()
    
    def load_presets(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    presets = json.load(f)
                logger.info(f"Đã tải presets từ {self.config_file}")
                return presets
            except Exception as e:
                logger.warning(f"Không thể tải presets: {e}, sử dụng presets mặc định")
        return self.DEFAULT_PRESETS.copy()
    
    def get_preset(self, preset_name): return self.presets.get(preset_name)
    def list_presets(self): return self.presets

class AudioNormalizer:
    """
    Lớp ứng dụng chính
    """
    def __init__(self):
        self.processor = AudioProcessor()
        self.preset_manager = PresetManager()

# --- GUI Application ---

class QueueHandler(logging.Handler):
    """Class to send logging records to a queue from different threads"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class DragDropEntry(ttk.Entry):
    """Custom Entry widget that supports drag and drop using tkinterdnd2"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.drop_target_register('DND_Files')
        self.dnd_bind('<<Drop>>', self._on_drop)

    def _on_drop(self, event):
        """Handle file/folder drop event"""
        path = event.data
        if "{" in path and "}" in path: # Tcl list format for paths with spaces
             path = path.strip('{}')

        if os.path.exists(path):
            folder_path = path if os.path.isdir(path) else os.path.dirname(path)
            self.delete(0, tk.END)
            self.insert(0, folder_path)
        return "break"

class NormalizerApp(TkinterDnD.Tk):
    """Main GUI Application class, inheriting from TkinterDnD.Tk"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.normalizer = AudioNormalizer()
        self.preset_keys = []

        self.title("Pink Noise Audio Normalizer")
        self.geometry("700x580")
        self.minsize(600, 480)

        # --- Variables ---
        self.input_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.use_spectral_var = tk.BooleanVar()

        # --- UI Setup ---
        self.setup_widgets()
        self.setup_logging()
        self._on_preset_selected() # Initialize checkbox based on default preset

    def on_closing(self):
        """Handle window closing to prevent errors on exit."""
        self.destroy()

    def setup_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        folder_frame = ttk.LabelFrame(main_frame, text="Folders", padding="10")
        folder_frame.pack(fill=tk.X, expand=False)
        folder_frame.columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.input_entry = DragDropEntry(folder_frame, textvariable=self.input_folder_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(folder_frame, text="Browse...", command=self._select_input_folder).grid(row=0, column=2, sticky="e", padx=5, pady=5)

        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.output_entry = DragDropEntry(folder_frame, textvariable=self.output_folder_var)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(folder_frame, text="Browse...", command=self._select_output_folder).grid(row=1, column=2, sticky="e", padx=5, pady=5)

        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, expand=False, pady=10)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Preset:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        presets = self.normalizer.preset_manager.list_presets()
        self.preset_keys = list(presets.keys())
        preset_display_names = [f"{key.upper()} ({p['target_lufs']} LUFS) - {p['name']}" for key, p in presets.items()]
        
        self.preset_combo = ttk.Combobox(settings_frame, values=preset_display_names, state="readonly")
        self.preset_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        if preset_display_names:
            self.preset_combo.current(4) # Default to 'streaming' preset
        
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self.spectral_check = ttk.Checkbutton(
            settings_frame,
            text="Use Spectral Matching (EQ to Pink Noise Profile)",
            variable=self.use_spectral_var
        )
        self.spectral_check.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.X, expand=False)
        action_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(action_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, pady=(0, 5))
        
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self._start_processing)
        self.start_button.pack(pady=5)

        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def _on_preset_selected(self, event=None):
        """Updates the spectral matching checkbox based on the selected preset's default."""
        selected_index = self.preset_combo.current()
        if selected_index != -1:
            preset_key = self.preset_keys[selected_index]
            preset = self.normalizer.preset_manager.get_preset(preset_key)
            self.use_spectral_var.set(preset.get('use_spectral_matching', True))

    def setup_logging(self):
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logger.addHandler(self.queue_handler)
        self.after(100, self._poll_log_queue)

    def _poll_log_queue(self):
        while True:
            try: record = self.log_queue.get(block=False)
            except queue.Empty: break
            else:
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, record + '\n')
                self.log_text.configure(state='disabled')
                self.log_text.yview(tk.END)
        self.after(100, self._poll_log_queue)
    
    def _select_input_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected: self.input_folder_var.set(folder_selected)

    def _select_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected: self.output_folder_var.set(folder_selected)

    def _start_processing(self):
        input_folder = self.input_folder_var.get()
        output_folder = self.output_folder_var.get()
        
        if not input_folder or not output_folder:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return

        selected_index = self.preset_combo.current()
        if selected_index == -1:
            messagebox.showerror("Error", "Please select a preset.")
            return
        preset_key = self.preset_keys[selected_index]
        use_spectral = self.use_spectral_var.get()
        
        self.start_button['state'] = 'disabled'
        self.progress['value'] = 0
        
        processing_thread = threading.Thread(
            target=self._process_batch_thread,
            args=(input_folder, output_folder, preset_key, use_spectral),
            daemon=True
        )
        processing_thread.start()
        
    def _process_batch_thread(self, input_folder, output_folder, preset_key, use_spectral):
        """The actual processing logic that runs in a background thread."""
        try:
            preset = self.normalizer.preset_manager.get_preset(preset_key)
            if not preset:
                logger.error(f"Preset '{preset_key}' does not exist.")
                return

            audio_files = self.normalizer.processor.find_audio_files(input_folder)
            if not audio_files:
                logger.warning("No audio files found in the selected folder.")
                return
                
            os.makedirs(output_folder, exist_ok=True)
            
            logger.info(f"Starting to process {len(audio_files)} files with preset '{preset['name']}' ({preset['target_lufs']} LUFS)")
            logger.info(f"Spectral matching: {'On' if use_spectral else 'Off'}")
            
            self.after(0, self._set_progress_max, len(audio_files))

            processed_count = 0
            for i, audio_file in enumerate(audio_files):
                logger.info(f"Processing file {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                success = self.normalizer.processor.process_single_file(
                    audio_file,
                    pink_reference_lufs=preset['target_lufs'],
                    output_folder=output_folder,
                    use_spectral_matching=use_spectral
                )
                if success:
                    processed_count += 1
                
                self.after(0, self._update_progress, i + 1)
            
            logger.info(f"Finished! Successfully processed {processed_count}/{len(audio_files)} files.")
            success_msg = f"Processing complete!\nSuccessfully processed {processed_count}/{len(audio_files)} files."
            self.after(0, lambda: messagebox.showinfo("Success", success_msg))

        except Exception as e:
            logger.error(f"A critical error occurred during processing: {e}", exc_info=True)
            error_msg = f"An error occurred: {e}"
            self.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.after(0, self._processing_finished)
            
    def _set_progress_max(self, max_value): self.progress['maximum'] = max_value
    def _update_progress(self, value): self.progress['value'] = value
    def _processing_finished(self): self.start_button['state'] = 'normal'

if __name__ == "__main__":
    app = NormalizerApp()
    app.mainloop()