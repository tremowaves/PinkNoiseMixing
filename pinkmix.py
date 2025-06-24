#!/usr/bin/env python3
"""
Advanced Keyword-Based Audio Normalizer (Game-Ready, Flexible CSV)

- Applies a professional audio processing chain: Filters -> Gate -> Compressor -> Normalization -> Limiter -> Fades.
- Supports both simple and advanced CSV formats by treating compressor/limiter columns as optional.
- Implements a more powerful noise gate using a true expansion ratio.
- Selects rules based on a priority system.

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
FADE_DURATION_MS = 5

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)


# --- Core Logic ---

class PinkNoiseGenerator:
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
    def __init__(self) -> None:
        self.rules: List[Dict[str, Any]] = []

    def load_rules_from_csv(self, csv_path: str) -> Tuple[bool, str]:
        try:
            df = pd.read_csv(csv_path)
            # --- Flexible Column Handling ---
            # Define truly required columns
            required_cols = {'priority', 'keywords', 'target_lufs', 'use_spectral_matching', 'lowcut', 'highcut', 'gate_threshold_db', 'expansion_ratio'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Rules CSV is missing required columns: {missing}")

            # Define optional columns and their defaults (which disable the effect)
            optional_cols = {
                'compressor_threshold_db': 0.0,
                'compressor_ratio': 1.0,
                'compressor_attack_ms': 5.0,
                'compressor_release_ms': 100.0,
                'limiter_threshold_db': -0.1
            }
            for col, default_val in optional_cols.items():
                if col not in df.columns:
                    logger.info(f"'{col}' not found in CSV, using default value: {default_val}")
                    df[col] = default_val

            df = df.sort_values(by='priority').reset_index(drop=True)
            df['keywords'] = df['keywords'].str.lower().str.split(',')
            self.rules = df.to_dict('records')
            logger.info(f"Successfully loaded and sorted {len(self.rules)} rules from {csv_path}")
            return True, f"Loaded {len(self.rules)} rules."
        except Exception as e:
            logger.error(f"Failed to load or parse Rules CSV: {e}", exc_info=True)
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
    def find_audio_files(self, folder_path: str) -> List[str]:
        audio_files = [os.path.join(r, f) for r, _, files in os.walk(folder_path) for f in files if f.lower().endswith(tuple(SUPPORTED_AUDIO_FORMATS))]
        logger.info(f"Found {len(audio_files)} audio files in input folder.")
        return audio_files

    def _apply_fades(self, audio: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        fade_len = int((fade_ms / 1000.0) * sr)
        if fade_len <= 0 or fade_len * 2 > audio.shape[-1]: return audio
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        if audio.ndim == 1:
            audio[:fade_len] *= fade_in; audio[-fade_len:] *= fade_in[::-1]
        else:
            audio[:, :fade_len] *= fade_in; audio[:, -fade_len:] *= fade_in[::-1]
        return audio

    def _apply_filters(self, audio: np.ndarray, sr: int, lowcut: float, highcut: float) -> np.ndarray:
        if lowcut >= highcut or (lowcut < 20 and highcut > sr / 2 - 100): return audio
        try:
            sos = signal.butter(5, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
            return signal.sosfilt(sos, audio)
        except Exception as e: logger.error(f"Filter failed: {e}"); return audio

    def _apply_dynamics_processor(self, audio: np.ndarray, sr: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, is_expander: bool = False) -> np.ndarray:
        if ratio == 1.0: return audio # No processing needed

        # Envelope detection
        abs_audio = np.abs(audio)
        attack_coeff = np.exp(-1.0 / (sr * (attack_ms / 1000.0)))
        release_coeff = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
        envelope = np.zeros_like(abs_audio)
        envelope[0] = abs_audio[0]
        for i in range(1, len(abs_audio)):
            coeff = attack_coeff if abs_audio[i] > envelope[i-1] else release_coeff
            envelope[i] = coeff * envelope[i-1] + (1 - coeff) * abs_audio[i]
        
        # Gain calculation in dB
        with np.errstate(divide='ignore'): # Ignore log10(0)
            envelope_db = 20 * np.log10(envelope)
        
        gain_db = np.zeros_like(envelope_db)
        if is_expander: # Downward expander / gate
            mask = envelope_db < threshold_db
            gain_db[mask] = (envelope_db[mask] - threshold_db) * (ratio - 1.0)
        else: # Compressor
            mask = envelope_db > threshold_db
            gain_db[mask] = (envelope_db[mask] - threshold_db) * (1.0 / ratio - 1.0)

        # Convert gain to linear and apply
        gain_lin = 10 ** (gain_db / 20.0)
        return audio * gain_lin

    def _analyze_frequency_spectrum(self, audio: np.ndarray, sr: int, n_fft: int = 2048) -> np.ndarray:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//4)
        return np.mean(librosa.amplitude_to_db(np.abs(stft), ref=np.max), axis=1)

    def _apply_spectral_adjustment(self, audio: np.ndarray, sr: int, pink_ref: np.ndarray) -> np.ndarray:
        audio_spectrum = self._analyze_frequency_spectrum(audio, sr)
        pink_spectrum = self._analyze_frequency_spectrum(pink_ref, sr)
        gain_diff = pink_spectrum - audio_spectrum
        gain_adjustments = gaussian_filter1d(gain_diff, sigma=2)
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        gain_linear = 10 ** (gain_adjustments / 20)
        stft_adjusted = stft * gain_linear.reshape(-1, 1)
        return librosa.istft(stft_adjusted, hop_length=512, length=len(audio))

    def _match_loudness_to_target(self, audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        meter = pyln.Meter(sr)
        try:
            current_lufs = meter.integrated_loudness(audio)
            if np.isinf(current_lufs): return audio
            gain_db = target_lufs - current_lufs
            return audio * (10 ** (gain_db / 20))
        except Exception as e: logger.error(f"Loudness match failed: {e}"); return audio

    def _process_channel(self, audio_channel: np.ndarray, sr: int, config: Dict[str, Any]) -> np.ndarray:
        # 1. Filters
        processed = self._apply_filters(audio_channel, sr, float(config['lowcut']), float(config['highcut']))
        # 2. Noise Gate (Downward Expander)
        processed = self._apply_dynamics_processor(processed, sr, float(config['gate_threshold_db']), float(config['expansion_ratio']), 5.0, 100.0, is_expander=True)
        # 3. Compressor
        processed = self._apply_dynamics_processor(processed, sr, float(config['compressor_threshold_db']), float(config['compressor_ratio']), float(config['compressor_attack_ms']), float(config['compressor_release_ms']))
        # 4. Normalization (Spectral or LUFS)
        if config['use_spectral_matching'] and (len(processed) / sr >= 0.5):
            pink_ref = PinkNoiseGenerator.generate_with_target_lufs(len(processed) / sr, sr, float(config['target_lufs']))
            if pink_ref.size > 0: processed = self._apply_spectral_adjustment(processed, sr, pink_ref)
        else:
            processed = self._match_loudness_to_target(processed, sr, float(config['target_lufs']))
        # 5. Limiter (Fast Compressor)
        processed = self._apply_dynamics_processor(processed, sr, float(config['limiter_threshold_db']), 100.0, 0.1, 5.0)
        return processed

    def process_file(self, audio_file: str, output_folder: str, norm_config: Dict[str, Any]) -> bool:
        try:
            data, sr = librosa.load(audio_file, sr=None, mono=False)
            if data.ndim == 2 and data.shape[0] > 0:
                processed_audio = np.array([self._process_channel(data[ch], sr, norm_config) for ch in range(data.shape[0])])
            elif data.ndim == 1:
                processed_audio = self._process_channel(data, sr, norm_config)
            else:
                logger.warning(f"Skipping empty audio file: {audio_file}"); return False
            # 6. Final Fades
            processed_audio = self._apply_fades(processed_audio, sr, FADE_DURATION_MS)
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(output_folder, f"processed_{base_name}.wav")
            sf.write(output_path, processed_audio.T if processed_audio.ndim == 2 else processed_audio, sr)
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
        self.title("Game Audio Processor"); self.geometry("800x950"); self.minsize(750, 900)
        self.input_folder_var, self.output_folder_var = tk.StringVar(), tk.StringVar()
        self.rules_csv_path_var, self.unmatched_folder_var = tk.StringVar(), tk.StringVar()
        self._setup_widgets(); self._setup_logging_queue()

    def _setup_widgets(self) -> None:
        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        path_frame = ttk.LabelFrame(main_frame, text="Step 1: Paths", padding="10")
        path_frame.pack(fill=tk.X, pady=5); path_frame.columnconfigure(1, weight=1)
        self._create_path_row(path_frame, "Input Folder:", 0, self.input_folder_var, True)
        self._create_path_row(path_frame, "Output Folder:", 1, self.output_folder_var, True)
        self._create_path_row(path_frame, "Rules CSV:", 2, self.rules_csv_path_var, False)
        unmatched_frame = ttk.LabelFrame(main_frame, text="Step 2: Unmatched File Handling", padding="10")
        unmatched_frame.pack(fill=tk.X, pady=5); unmatched_frame.columnconfigure(1, weight=1)
        self._create_path_row(unmatched_frame, "Move Unmatched To (Optional):", 0, self.unmatched_folder_var, True)
        fallback_frame = ttk.LabelFrame(main_frame, text="Step 3: Fallback Settings (for unmatched files)", padding="10")
        fallback_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self._create_fallback_widgets(fallback_frame)
        action_frame = ttk.LabelFrame(main_frame, text="Step 4: Process", padding="10")
        action_frame.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(action_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, pady=5)
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self._start_processing_clicked, style="Accent.TButton")
        self.start_button.pack(pady=5); ttk.Style(self).configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=10, wrap=tk.WORD, bg="#f0f0f0")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _create_fallback_widgets(self, parent: ttk.Frame):
        parent.columnconfigure(1, weight=1)
        self.fb_vars = {
            'lufs': tk.DoubleVar(value=-14.0), 'spectral': tk.BooleanVar(value=False),
            'lowcut': tk.DoubleVar(value=80.0), 'highcut': tk.DoubleVar(value=12000.0),
            'gate_thresh': tk.DoubleVar(value=-50.0), 'exp_ratio': tk.DoubleVar(value=10.0),
            'comp_thresh': tk.DoubleVar(value=-20.0), 'comp_ratio': tk.DoubleVar(value=4.0),
            'comp_attack': tk.DoubleVar(value=5.0), 'comp_release': tk.DoubleVar(value=100.0),
            'lim_thresh': tk.DoubleVar(value=-1.0)
        }
        norm_frame = ttk.LabelFrame(parent, text="Normalization", padding=5); norm_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5, columnspan=2); norm_frame.columnconfigure(1, weight=1)
        filter_frame = ttk.LabelFrame(parent, text="Filters & Gate", padding=5); filter_frame.grid(row=1, column=0, sticky="ewns", padx=5, pady=5); filter_frame.columnconfigure(1, weight=1)
        dyn_frame = ttk.LabelFrame(parent, text="Dynamics", padding=5); dyn_frame.grid(row=1, column=1, sticky="ewns", padx=5, pady=5); dyn_frame.columnconfigure(1, weight=1)
        self._add_spinbox(norm_frame, "Target LUFS:", 0, self.fb_vars['lufs'], (-60, 0, 0.5))
        ttk.Checkbutton(norm_frame, text="Use Spectral Matching", variable=self.fb_vars['spectral']).grid(row=0, column=2, sticky='w', padx=10)
        self._add_spinbox(filter_frame, "Low Cut (Hz):", 0, self.fb_vars['lowcut'], (20, 20000, 10))
        self._add_spinbox(filter_frame, "High Cut (Hz):", 1, self.fb_vars['highcut'], (100, 22000, 100))
        self._add_spinbox(filter_frame, "Gate Thresh (dB):", 2, self.fb_vars['gate_thresh'], (-90, 0, 1))
        self._add_spinbox(filter_frame, "Exp. Ratio (1:n):", 3, self.fb_vars['exp_ratio'], (1.0, 100.0, 0.1))
        self._add_spinbox(dyn_frame, "Comp Thresh (dB):", 0, self.fb_vars['comp_thresh'], (-60, 0, 1))
        self._add_spinbox(dyn_frame, "Comp Ratio (n:1):", 1, self.fb_vars['comp_ratio'], (1.0, 20.0, 0.1))
        self._add_spinbox(dyn_frame, "Attack (ms):", 2, self.fb_vars['comp_attack'], (0.1, 100, 0.1))
        self._add_spinbox(dyn_frame, "Release (ms):", 3, self.fb_vars['comp_release'], (10, 1000, 10))
        self._add_spinbox(dyn_frame, "Limiter (dBFS):", 4, self.fb_vars['lim_thresh'], (-12, 0, 0.1))

    def _add_spinbox(self, parent, label, row, var, range_):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Spinbox(parent, from_=range_[0], to=range_[1], increment=range_[2], textvariable=var, width=10).grid(row=row, column=1, sticky='ew', padx=5, pady=2)

    def _create_path_row(self, p, txt, r, var, is_dir):
        ttk.Label(p, text=txt).grid(row=r, column=0, sticky="w", padx=5, pady=5)
        DragDropEntry(p, textvariable=var).grid(row=r, column=1, sticky="ew")
        cmd=lambda:var.set(filedialog.askdirectory(title=f"Select {txt}")) if is_dir else var.set(filedialog.askopenfilename(title="Select File",filetypes=[("CSV","*.csv")]))
        ttk.Button(p, text="Browse...", command=cmd).grid(row=r, column=2, padx=5)

    def _setup_logging_queue(self): self.log_queue = queue.Queue(); logger.addHandler(QueueHandler(self.log_queue)); self.after(100, self._poll_log_queue)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            try:
                rec = self.log_queue.get(block=False)
                self.log_text.configure(state='normal'); self.log_text.insert(tk.END, rec + '\n')
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
                'target_lufs': self.fb_vars['lufs'].get(), 'use_spectral_matching': self.fb_vars['spectral'].get(),
                'lowcut': self.fb_vars['lowcut'].get(), 'highcut': self.fb_vars['highcut'].get(),
                'gate_threshold_db': self.fb_vars['gate_thresh'].get(), 'expansion_ratio': self.fb_vars['exp_ratio'].get(),
                'compressor_threshold_db': self.fb_vars['comp_thresh'].get(), 'compressor_ratio': self.fb_vars['comp_ratio'].get(),
                'compressor_attack_ms': self.fb_vars['comp_attack'].get(), 'compressor_release_ms': self.fb_vars['comp_release'].get(),
                'limiter_threshold_db': self.fb_vars['lim_thresh'].get()
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
                rule = self.rule_manager.get_rule_for_filename(basename)
                if rule:
                    if self.processor.process_file(audio_file, output_folder, rule): processed += 1
                    else: failed += 1
                elif unmatched_folder:
                    try: shutil.move(audio_file, os.path.join(unmatched_folder, basename)); moved += 1; logger.info(f"Moved unmatched file: {basename}")
                    except Exception as e: failed += 1; logger.error(f"Failed to move {basename}: {e}")
                else:
                    if self.processor.process_file(audio_file, output_folder, fallback_config): processed += 1
                    else: failed += 1
                self.after(0, self.progress.config, {'value': i})
            
            summary = f"Processing complete!\n\nProcessed: {processed}\nMoved: {moved}\nFailed: {failed}"
            logger.info(summary); self.after(0, lambda: messagebox.showinfo("Success", summary))

        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}"))
        finally:
            self.after(0, lambda: self.start_button.config(state='normal'))

if __name__ == "__main__":
    app = NormalizerApp()
    app.mainloop()