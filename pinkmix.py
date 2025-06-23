#!/usr/bin/env python3
"""
Pink Noise Audio Normalizer
Ứng dụng cân bằng âm thanh với Pink Noise Reference
Hỗ trợ nhiều reference level cho các loại audio khác nhau
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

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pink_noise_normalizer.log', mode='w'), # Overwrite log each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PinkNoiseGenerator:
    """
    Generator tạo Pink Noise với các thuật toán khác nhau
    """
    
    @staticmethod
    def generate_frequency_domain(n_samples, sample_rate):
        """
        Tạo pink noise trong miền tần số
        """
        white_noise = np.random.randn(n_samples)
        white_fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples, d=1/sample_rate)
        
        # Áp dụng scaling factor 1/√f cho pink noise, tránh chia cho 0
        scale = np.ones_like(freqs)
        if freqs[0] == 0:
            scale[1:] = 1 / np.sqrt(freqs[1:])
        else: # Should not happen with rfftfreq
            scale = 1 / np.sqrt(freqs)

        pink_fft = white_fft * scale
        pink_noise = np.fft.irfft(pink_fft, n=n_samples)
        
        return pink_noise

    @staticmethod
    def generate_with_target_lufs(duration_seconds, sample_rate, target_lufs):
        """
        Tạo pink noise với target LUFS level cụ thể
        """
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
        """
        Tìm tất cả file audio trong thư mục và thư mục con
        """
        audio_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_formats):
                    full_path = os.path.join(root, file)
                    audio_files.append(full_path)
        logger.info(f"Tìm thấy {len(audio_files)} file audio")
        return audio_files
    
    def analyze_frequency_spectrum(self, audio, sr, n_fft=2048):
        """
        Phân tích phổ tần số của audio
        """
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//4)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        avg_spectrum = np.mean(magnitude_db, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return freqs, avg_spectrum
    
    def calculate_gain_adjustments(self, audio_spectrum, pink_spectrum):
        """
        Tính toán gain adjustments cần thiết. Chỉ attenuate, không boost.
        """
        # Hiệu số: những tần số trong audio lớn hơn trong pink noise sẽ dương
        gain_diff = audio_spectrum - pink_spectrum
        # Chỉ attenuate (giảm) các tần số vượt quá pink noise (chỉ áp dụng gain âm)
        gain_adjustments = np.where(gain_diff > 0, -gain_diff, 0)
        # Làm mượt đường cong gain để tránh artifact
        gain_adjustments = gaussian_filter1d(gain_adjustments, sigma=2)
        return gain_adjustments
    
    def apply_spectral_adjustment(self, audio, sr, gain_adjustments, n_fft=2048):
        """
        Áp dụng spectral adjustment
        """
        hop_length = n_fft // 4
        audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')
        stft = librosa.stft(audio_padded, n_fft=n_fft, hop_length=hop_length)
        
        gain_linear = 10 ** (gain_adjustments / 20)
        gain_linear = gain_linear.reshape(-1, 1)  # Broadcast để match STFT shape
        
        stft_adjusted = stft * gain_linear
        
        # Thêm `length` để đảm bảo độ dài khớp với audio đã pad
        audio_adjusted = librosa.istft(stft_adjusted, hop_length=hop_length, length=len(audio_padded))
        
        # Cắt bỏ padding
        audio_adjusted = audio_adjusted[n_fft // 2 : len(audio) + n_fft // 2]
        return audio_adjusted
    
    def match_loudness_to_target(self, audio, sr, target_lufs):
        """
        Match loudness của audio với target LUFS
        """
        meter = pyln.Meter(sr)
        try:
            current_lufs = meter.integrated_loudness(audio)
            if np.isinf(current_lufs) or np.isnan(current_lufs):
                logger.warning("Audio quá nhỏ để đo LUFS, bỏ qua loudness matching.")
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
            logger.error(f"Lỗi khi match loudness: {e}")
            return audio
    
    def _process_channel(self, audio_channel, sr, pink_reference_lufs, use_spectral_matching):
        """
        Xử lý một channel audio
        """
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
        """
        Xử lý một file audio
        """
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
            
            # soundfile mong đợi shape (samples, channels) nên cần chuyển vị .T
            sf.write(output_path, processed_audio.T if processed_audio.ndim == 2 else processed_audio, sr)
            return True
        except Exception as e:
            logger.error(f"Lỗi xử lý {audio_file}: {e}")
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
    
    def save_presets(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False)
            logger.info(f"Đã lưu presets vào {self.config_file}")
        except Exception as e:
            logger.error(f"Không thể lưu presets: {e}")
    
    def get_preset(self, preset_name): return self.presets.get(preset_name)
    def list_presets(self): return self.presets
    
    def add_custom_preset(self, name, target_lufs, description="Custom preset", use_spectral_matching=True):
        self.presets[name] = {
            'name': name.title(), 'target_lufs': target_lufs,
            'description': description, 'use_spectral_matching': use_spectral_matching
        }
        self.save_presets()

class AudioNormalizer:
    """
    Lớp ứng dụng chính
    """
    def __init__(self):
        self.processor = AudioProcessor()
        self.preset_manager = PresetManager()
    
    def process_batch(self, input_folder, output_folder, preset_name='streaming', n_workers=None):
        preset = self.preset_manager.get_preset(preset_name)
        if not preset: raise ValueError(f"Preset '{preset_name}' không tồn tại")
        
        audio_files = self.processor.find_audio_files(input_folder)
        if not audio_files:
            logger.warning("Không tìm thấy file audio nào")
            return 0
        
        os.makedirs(output_folder, exist_ok=True)
        
        if n_workers is None: n_workers = max(1, mp.cpu_count() - 1)
        
        logger.info(f"Bắt đầu xử lý {len(audio_files)} file với preset '{preset['name']}' ({preset['target_lufs']} LUFS)")
        logger.info(f"Spectral matching: {'On' if preset['use_spectral_matching'] else 'Off'}")
        
        process_func = partial(self.processor.process_single_file, pink_reference_lufs=preset['target_lufs'],
                               output_folder=output_folder, use_spectral_matching=preset['use_spectral_matching'])
        
        processed_count = 0
        with mp.Pool(n_workers) as pool:
            with tqdm(total=len(audio_files), desc="Processing") as pbar:
                for success in pool.imap_unordered(process_func, audio_files):
                    if success: processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"Success": processed_count})
        
        logger.info(f"Hoàn thành! Đã xử lý thành công {processed_count}/{len(audio_files)} file")
        return processed_count
    
    def generate_test_tones(self, output_folder, duration=10):
        os.makedirs(output_folder, exist_ok=True)
        for preset_name, preset in self.preset_manager.list_presets().items():
            pink_noise = PinkNoiseGenerator.generate_with_target_lufs(duration, 48000, preset['target_lufs'])
            filename = f"pink_noise_{preset_name}_{preset['target_lufs']}LUFS.wav"
            sf.write(os.path.join(output_folder, filename), pink_noise, 48000)
            logger.info(f"Đã tạo test tone: {filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Pink Noise Audio Normalizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_folder output_folder --preset music
  %(prog)s input_folder output_folder --preset sfx --workers 4
  %(prog)s --list-presets
  %(prog)s --generate-tones test_tones/
  %(prog)s --add-preset "My Preset" -15 "A new preset" --spectral-mode off
        """
    )
    
    parser.add_argument('input_folder', nargs='?', help='Thư mục chứa file audio input')
    parser.add_argument('output_folder', nargs='?', help='Thư mục lưu file đã xử lý')
    parser.add_argument('--preset', default='streaming', help='Preset để sử dụng (default: streaming)')
    parser.add_argument('--workers', type=int, help='Số workers cho parallel processing')
    parser.add_argument('--list-presets', action='store_true', help='Liệt kê tất cả presets có sẵn')
    parser.add_argument('--generate-tones', metavar='OUTPUT_DIR', help='Tạo test tones vào thư mục chỉ định')
    parser.add_argument('--add-preset', nargs=3, metavar=('NAME', 'LUFS', 'DESCRIPTION'), help='Thêm custom preset mới. Dùng chung với --spectral-mode.')
    parser.add_argument('--spectral-mode', choices=['on', 'off'], default='on', help="Bật/tắt spectral matching khi thêm preset (default: on)")
    
    args = parser.parse_args()
    normalizer = AudioNormalizer()
    
    if args.list_presets:
        print("\nCác presets có sẵn:")
        print("-" * 70)
        print(f"{'Name':<15} | {'Target LUFS':<12} | {'Spectral Matching':<20} | {'Description'}")
        print("-" * 70)
        for name, p in normalizer.preset_manager.list_presets().items():
            spec_match_str = 'On' if p.get('use_spectral_matching', False) else 'Off'
            print(f"{name:<15} | {p['target_lufs']:<12.1f} | {spec_match_str:<20} | {p['description']}")
        return
    
    if args.generate_tones:
        normalizer.generate_test_tones(args.generate_tones)
        return
    
    if args.add_preset:
        name, lufs_str, description = args.add_preset
        use_spectral = (args.spectral_mode == 'on')
        try:
            lufs = float(lufs_str)
            normalizer.preset_manager.add_custom_preset(name, lufs, description, use_spectral_matching=use_spectral)
            print(f"\n✅ Đã thêm preset '{name}'.")
        except ValueError:
            parser.error("Lỗi: Giá trị LUFS phải là một con số (ví dụ: -14.5).")
        return
    
    if not args.input_folder or not args.output_folder:
        parser.error("Lỗi: Cần chỉ định input_folder và output_folder cho việc xử lý.")
    if not os.path.isdir(args.input_folder):
        parser.error(f"Lỗi: Input folder không tồn tại: {args.input_folder}")
    if not normalizer.preset_manager.get_preset(args.preset):
        keys = ', '.join(normalizer.preset_manager.list_presets().keys())
        parser.error(f"Lỗi: Preset '{args.preset}' không tồn tại. Các preset có sẵn: {keys}")
    
    try:
        count = normalizer.process_batch(args.input_folder, args.output_folder, args.preset, args.workers)
        if count > 0: print(f"\n✅ Hoàn thành! Output folder: {os.path.abspath(args.output_folder)}")
        else: print("\n⚠️  Không có file nào được xử lý thành công.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong quá trình xử lý: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()