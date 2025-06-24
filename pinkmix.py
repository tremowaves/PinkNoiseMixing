#!/usr/bin/env python3
"""
Advanced Keyword-Based Audio Normalizer (Corrected Logic)

- Normalizes audio files based on keyword matching from a rules CSV.
- Applies a chain of effects: Filters, Noise Gate, and a final processing stage.
- Final stage is EITHER Spectral Matching OR Loudness Normalization, preventing loudness homogenization.
- Selects rules based on a priority system.
- Moves any files that do not match a rule to a specified folder.

GUI Version
"""

# --- Standard Library Imports ---
import logging
import os
import queue
import shutil
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

# --- Third-Party Imports ---
try:
    import librosa
    import numpy as np
    import pandas as pd
    import pyloudnorm as pyln
    import soundfile as sf
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
except ImportError as e:
    print(f"Error: A required library is missing: {e}", file=sys.stderr)
    print("Please install the necessary libraries by running:", file=sys.stderr)
    print("pip install librosa numpy pandas pyloudnorm soundfile scipy", file=sys.stderr)
    sys.exit(1)

# --- GUI Specific Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
try:
    from tkinterdnd2 import TkinterDnD
except ImportError:
    print("Error: tkinterdnd2 library not found.", file=sys.stderr)
    print("Please install it using: pip install tkinterdnd2-py", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg']
LOG_FILE_NAME = 'audio_normalizer.log'
FADE_DURATION_MS = 5 # Duration for fade-in and fade-out in milliseconds to prevent clicks.

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# --- Core Logic ---

class PinkNoiseGenerator:
    """Generates Pink Noise, which has equal energy per octave."""
    @staticmethod
    def generate_with_target_lufs(duration_seconds: float, sample_rate: int, target_lufs: float) -> np.ndarray:
        n_samples = int(duration_seconds * sample_rate)
        if n_samples <= 0: return np.array([])
        white_noise = np.random.randn(n_samples)
        white_fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples, d=1/sample_rate)
        scale = np.ones_like(freqs)
        non_zero_freqs = freqs > 0
        scale[non_zero_freqs] = 1 / np.sqrt(freqs[non_zero_freqs])
        pink_fft = white_fft * scale
        pink_noise = np.fft.irfft(pink_fft, n=n_samples)
        meter = pyln.Meter(sample_rate)
        current_lufs = meter.integrated_loudness(pink_noise)
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        pink_noise_normalized = pink_noise * gain_linear
        if np.max(np.abs(pink_noise_normalized)) > 0.98:
            pink_noise_normalized *= (0.98 / np.max(np.abs(pink_noise_normalized)))
        return pink_noise_normalized

class RuleManager:
    """Manages normalization rules loaded and sorted from a CSV file."""
    def __init__(self) -> None:
        self.rules: List[Dict[str, Any]] = []

    def load_rules_from_csv(self, csv_path: str) -> Tuple[bool, str]:
        try:
            df = pd.read_csv(csv_path)
            required_cols = {'priority', 'keywords', 'target_lufs', 'use_spectral_matching', 'lowcut', 'highcut', 'gate_threshold_db', 'expansion_ratio'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Rules CSV is missing required columns: {missing}")
            df = df.sort_values(by='priority').reset_index(drop=True)
            df['keywords'] = df['keywords'].str.lower().str.split(',')
            self.rules = df.to_dict('records')
            logger.info(f"Successfully loaded and sorted {len(self.rules)} rules from {csv_path}")
            return True, f"Loaded {len(self.rules)} rules."
        except Exception as e:
            logger.error(f"Failed to load or parse Rules CSV: {e}")
            self.rules = []
            return False, str(e)

    def get_rule_for_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        fn_lower = filename.lower()
        for rule in self.rules:
            for keyword in rule.get('keywords', []):
                if keyword.strip() in fn_lower:
                    logger.info(f"Rule match for '{filename}' on keyword '{keyword}' (Priority: {rule['priority']}, Category: {rule.get('category_name', 'N/A')}).")
                    return rule
        return None

class AudioProcessor:
    """Handles all audio file operations: discovery, processing, and saving."""
    def find_audio_files(self, folder_path: str) -> List[str]:
        audio_files = [os.path.join(r, f) for r, _, files in os.walk(folder_path) for f in files if f.lower().endswith(tuple(SUPPORTED_AUDIO_FORMATS))]
        logger.info(f"Found {len(audio_files)} audio files in input folder.")
        return audio_files

    def _apply_fades(self, audio: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        fade_len_samples = int((fade_ms / 1000.0) * sr)
        audio_len_samples = audio.shape[-1]
        if fade_len_samples <= 0: return audio
        if fade_len_samples * 2 > audio_len_samples: fade_len_samples = audio_len_samples // 2
        if fade_len_samples == 0: return audio
        fade_in = np.linspace(0.0, 1.0, fade_len_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len_samples, dtype=np.float32)
        if audio.ndim == 1:
            audio[:fade_len_samples] *= fade_in; audio[-fade_len_samples:] *= fade_out
        elif audio.ndim == 2:
            audio[:, :fade_len_samples] *= fade_in; audio[:, -fade_len_samples:] *= fade_out
        return audio

    def _apply_filters(self, audio: np.ndarray, sr: int, lowcut: float, highcut: float) -> np.ndarray:
        if lowcut >= highcut or (lowcut < 20 and highcut > sr / 2 - 100): return audio
        try:
            sos = signal.butter(5, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
            return signal.sosfilt(sos, audio)
        except Exception as e: logger.error(f"Could not apply filter: {e}"); return audio

    def _apply_noise_gate(self, audio: np.ndarray, sr: int, threshold_db: float, ratio: float) -> np.ndarray:
        if threshold_db >= 0 or not (0 < ratio <= 1): return audio
        threshold_lin = 10.0 ** (threshold_db / 20.0)
        frame_len = int(0.01 * sr); hop_len = int(0.005 * sr)
        rms_frames = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        gain = np.ones_like(rms_frames)
        below_thresh = rms_frames < threshold_lin
        gain[below_thresh] = ((rms_frames[below_thresh] / threshold_lin) - 1.0) * (1.0 - ratio) + 1.0
        gain_interpolated = np.interp(np.arange(len(audio)), np.arange(len(gain)) * hop_len, gain)
        return audio * gain_interpolated

    def _analyze_frequency_spectrum(self, audio: np.ndarray, sr: int, n_fft: int = 2048) -> np.ndarray:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//4)
        return np.mean(librosa.amplitude_to_db(np.abs(stft), ref=np.max), axis=1)

    def _apply_spectral_adjustment(self, audio: np.ndarray, sr: int, pink_ref: np.ndarray, n_fft: int = 2048) -> np.ndarray:
        audio_spectrum = self._analyze_frequency_spectrum(audio, sr, n_fft)
        pink_spectrum = self._analyze_frequency_spectrum(pink_ref, sr, n_fft)
        gain_diff = pink_spectrum - audio_spectrum
        gain_adjustments = gaussian_filter1d(gain_diff, sigma=2)
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//4)
        gain_linear = 10 ** (gain_adjustments / 20)
        stft_adjusted = stft * gain_linear.reshape(-1, 1)
        return librosa.istft(stft_adjusted, hop_length=n_fft//4, length=len(audio))

    def _match_loudness_to_target(self, audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        meter = pyln.Meter(sr)
        try:
            current_lufs = meter.integrated_loudness(audio)
            if np.isinf(current_lufs) or np.isnan(current_lufs):
                logger.warning("Audio is silent/short; skipping loudness matching.")
                return audio
            gain_db = target_lufs - current_lufs
            audio_normalized = audio * (10 ** (gain_db / 20))
            if np.max(np.abs(audio_normalized)) > 0.98:
                audio_normalized *= (0.98 / np.max(np.abs(audio_normalized)))
                logger.warning("Applied peak limiting after loudness normalization.")
            return audio_normalized
        except ValueError as e: logger.warning(f"Could not measure LUFS: {e}. Skipping.")
        except Exception as e: logger.error(f"Error during loudness matching: {e}", exc_info=True)
        return audio

    def _process_channel(self, audio_channel: np.ndarray, sr: int, norm_config: Dict[str, Any]) -> np.ndarray:
        processed_audio = audio_channel
        processed_audio = self._apply_filters(processed_audio, sr, float(norm_config['lowcut']), float(norm_config['highcut']))
        processed_audio = self._apply_noise_gate(processed_audio, sr, float(norm_config['gate_threshold_db']), float(norm_config['expansion_ratio']))
        
        # --- CORRECTED LOGIC: Perform EITHER spectral matching OR loudness normalization ---
        if norm_config['use_spectral_matching'] and (len(processed_audio) / sr >= 0.5):
            logger.info(f"Applying spectral match to target LUFS: {norm_config['target_lufs']}")
            pink_ref = PinkNoiseGenerator.generate_with_target_lufs(len(processed_audio) / sr, sr, float(norm_config['target_lufs']))
            if pink_ref.size > 0:
                processed_audio = self._apply_spectral_adjustment(processed_audio, sr, pink_ref)
        else:
            logger.info(f"Applying loudness normalization to target LUFS: {norm_config['target_lufs']}")
            processed_audio = self._match_loudness_to_target(processed_audio, sr, float(norm_config['target_lufs']))
        
        return processed_audio

    def process_file(self, audio_file: str, output_folder: str, norm_config: Dict[str, Any]) -> bool:
        try:
            data, sr = librosa.load(audio_file, sr=None, mono=False)
            if data.ndim == 2 and data.shape[0] > 0:
                processed_channels = [self._process_channel(data[ch], sr, norm_config) for ch in range(data.shape[0])]
                processed_audio = np.array(processed_channels)
            elif data.ndim == 1:
                processed_audio = self._process_channel(data, sr, norm_config)
            else:
                logger.warning(f"Skipping empty audio file: {audio_file}"); return False

            processed_audio = self._apply_fades(processed_audio, sr, FADE_DURATION_MS)

            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(output_folder, f"processed_{base_name}.wav")
            output_data = processed_audio.T if processed_audio.ndim == 2 else processed_audio
            sf.write(output_path, output_data, sr)
            return True
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}", exc_info=True)
            return False

# --- GUI Application ---

class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record: logging.LogRecord) -> None: self.log_queue.put(self.format(record))

class DragDropEntry(ttk.Entry):
    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.drop_target_register('DND_Files'); self.dnd_bind('<<Drop>>', self.on_drop)
    def on_drop(self, event: tk.Event) -> None:
        path = event.data.strip('{}'); self.delete(0, tk.END); self.insert(0, path)

class NormalizerApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.processor = AudioProcessor(); self.rule_manager = RuleManager()
        self.title("Advanced Audio Normalizer"); self.geometry("750x850"); self.minsize(700, 800)
        self.input_folder_var = tk.StringVar(); self.output_folder_var = tk.StringVar()
        self.rules_csv_path_var = tk.StringVar(); self.unmatched_folder_var = tk.StringVar()
        self._setup_widgets(); self._setup_logging_queue()

    def _setup_widgets(self) -> None:
        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(main_frame, text="Step 1: Select Folders & Rules File", padding="10")
        config_frame.pack(fill=tk.X, expand=False, pady=5); config_frame.columnconfigure(1, weight=1)
        self._create_path_entry_row(config_frame, "Input Folder:", 0, self.input_folder_var, True)
        self._create_path_entry_row(config_frame, "Output Folder:", 1, self.output_folder_var, True)
        self._create_path_entry_row(config_frame, "Rules CSV:", 2, self.rules_csv_path_var, False)
        unmatched_frame = ttk.LabelFrame(main_frame, text="Step 2: Handling Unmatched Files", padding="10")
        unmatched_frame.pack(fill=tk.X, expand=False, pady=5); unmatched_frame.columnconfigure(1, weight=1)
        self._create_path_entry_row(unmatched_frame, "Move Unmatched To (Optional):", 0, self.unmatched_folder_var, True)
        fallback_frame = ttk.LabelFrame(main_frame, text="Step 3: Fallback Settings (for unmatched files)", padding="10")
        fallback_frame.pack(fill=tk.X, expand=False, pady=5)
        self._create_fallback_widgets(fallback_frame)
        action_frame = ttk.LabelFrame(main_frame, text="Step 4: Process Files", padding="10")
        action_frame.pack(fill=tk.X, expand=False, pady=5)
        self.progress = ttk.Progressbar(action_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, pady=5)
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self._start_processing_clicked, style="Accent.TButton")
        self.start_button.pack(pady=5); ttk.Style(self).configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=10, wrap=tk.WORD, bg="#f0f0f0")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def _create_fallback_widgets(self, parent: ttk.Frame):
        parent.columnconfigure(1, weight=1); parent.columnconfigure(3, weight=1)
        self.fallback_lufs_var = tk.DoubleVar(value=-14.0); self.fallback_spectral_var = tk.BooleanVar(value=True)
        self.fallback_lowcut_var = tk.DoubleVar(value=80.0); self.fallback_highcut_var = tk.DoubleVar(value=12000.0)
        self.fallback_gate_thresh_var = tk.DoubleVar(value=-50.0); self.fallback_exp_ratio_var = tk.DoubleVar(value=0.1)
        ttk.Label(parent, text="Fallback LUFS:").grid(row=0, column=0, sticky='w', padx=5, pady=3)
        ttk.Spinbox(parent, from_=-40.0, to=0.0, increment=0.5, textvariable=self.fallback_lufs_var, width=10).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Checkbutton(parent, text="Use Spectral Matching", variable=self.fallback_spectral_var).grid(row=0, column=2, columnspan=2, sticky='w', padx=20)
        ttk.Label(parent, text="Low Cut (Hz):").grid(row=1, column=0, sticky='w', padx=5, pady=3)
        ttk.Spinbox(parent, from_=20, to=20000, increment=10, textvariable=self.fallback_lowcut_var, width=10).grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Label(parent, text="High Cut (Hz):").grid(row=1, column=2, sticky='w', padx=20, pady=3)
        ttk.Spinbox(parent, from_=100, to=22000, increment=100, textvariable=self.fallback_highcut_var, width=10).grid(row=1, column=3, sticky='ew', padx=5)
        ttk.Label(parent, text="Gate Threshold (dB):").grid(row=2, column=0, sticky='w', padx=5, pady=3)
        ttk.Spinbox(parent, from_=-90, to=0, increment=1, textvariable=self.fallback_gate_thresh_var, width=10).grid(row=2, column=1, sticky='ew', padx=5)
        ttk.Label(parent, text="Expansion Ratio:").grid(row=2, column=2, sticky='w', padx=20, pady=3)
        ttk.Spinbox(parent, from_=0.01, to=1.0, increment=0.01, textvariable=self.fallback_exp_ratio_var, width=10).grid(row=2, column=3, sticky='ew', padx=5)

    def _create_path_entry_row(self, parent: tk.Widget, label_text: str, row: int, variable: tk.StringVar, is_directory: bool):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        DragDropEntry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew")
        command = lambda: variable.set(filedialog.askdirectory(title=f"Select {label_text}")) if is_directory else variable.set(filedialog.askopenfilename(title="Select Rules File", filetypes=[("CSV files", "*.csv")]))
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, padx=5)

    def _setup_logging_queue(self): self.log_queue = queue.Queue(); logger.addHandler(QueueHandler(self.log_queue)); self.after(100, self._poll_log_queue)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get(block=False)
                self.log_text.configure(state='normal'); self.log_text.insert(tk.END, record + '\n')
                self.log_text.configure(state='disabled'); self.log_text.yview(tk.END)
            except queue.Empty: pass
        self.after(100, self._poll_log_queue)

    def _start_processing_clicked(self):
        paths = [self.input_folder_var.get(), self.output_folder_var.get(), self.rules_csv_path_var.get()]
        if not all(paths): messagebox.showerror("Error", "Please select input/output folders and a Rules CSV file."); return
        success, msg = self.rule_manager.load_rules_from_csv(paths[2])
        if not success: messagebox.showerror("CSV Error", f"Failed to load Rules CSV:\n{msg}"); return
        self.start_button.config(state='disabled'); self.progress['value'] = 0
        threading.Thread(target=self._process_batch_thread, daemon=True).start()

    def _process_batch_thread(self):
        input_folder, output_folder, unmatched_folder = self.input_folder_var.get(), self.output_folder_var.get(), self.unmatched_folder_var.get()
        try:
            if unmatched_folder: os.makedirs(unmatched_folder, exist_ok=True)
            fallback_config = {
                'target_lufs': self.fallback_lufs_var.get(), 'use_spectral_matching': self.fallback_spectral_var.get(),
                'lowcut': self.fallback_lowcut_var.get(), 'highcut': self.fallback_highcut_var.get(),
                'gate_threshold_db': self.fallback_gate_thresh_var.get(), 'expansion_ratio': self.fallback_exp_ratio_var.get()
            }
            audio_files = self.processor.find_audio_files(input_folder)
            if not audio_files:
                logger.warning("No audio files found."); self.after(0, lambda: messagebox.showinfo("Finished", "No audio files were found."))
                return

            os.makedirs(output_folder, exist_ok=True)
            self.after(0, self.progress.config, {'maximum': len(audio_files)})
            processed, moved, failed = 0, 0, 0
            
            for i, audio_file in enumerate(audio_files, 1):
                basename = os.path.basename(audio_file)
                norm_config = self.rule_manager.get_rule_for_filename(basename)
                if norm_config:
                    if self.processor.process_file(audio_file, output_folder, norm_config): processed += 1
                    else: failed += 1
                elif unmatched_folder:
                    try: shutil.move(audio_file, os.path.join(unmatched_folder, basename)); logger.info(f"No match for '{basename}'. Moved."); moved += 1
                    except Exception as e: logger.error(f"Failed to move '{basename}': {e}"); failed += 1
                else:
                    logger.info(f"--- Processing '{basename}' with fallback settings ---")
                    if self.processor.process_file(audio_file, output_folder, fallback_config): processed += 1
                    else: failed += 1
                self.after(0, self.progress.config, {'value': i})
            
            parts = ["Processing complete!"]
            if processed > 0: parts.append(f"Successfully processed {processed} file(s).")
            if moved > 0: parts.append(f"Moved {moved} unmatched file(s).")
            if failed > 0: parts.append(f"Failed to process or move {failed} file(s). See log for details.")
            summary = "\n".join(parts)
            logger.info(summary); self.after(0, lambda: messagebox.showinfo("Success", summary))
        except Exception as e:
            logger.error(f"A critical error occurred: {e}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}"))
        finally:
            self.after(0, lambda: self.start_button.config(state='normal'))

if __name__ == "__main__":
    app = NormalizerApp()
    app.mainloop()