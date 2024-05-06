import torchaudio
from numpy import array
from torch import Tensor
from typing import Tuple, Callable
from scipy.signal import find_peaks

def standard_scale(x: Tensor) -> Tensor:
    x_mean = x.mean()
    x_std = x.std() + 1e-10

    scaled = x.clone()
    scaled -= x_mean
    scaled /= x_std
    
    return scaled

def min_max_scale(x: Tensor) -> Tensor:
    x_min = x.min()
    x_max = x.max()

    scaled = x.clone()
    scaled -= x_min
    scaled /= x_max - x_min

    return scaled

class AlignAudios:
    def __init__(self) -> None:
        pass

    def load_audio(self, audio_dir: str) -> Tuple[Tensor, Tensor]:
        audio, sample_rate = torchaudio.load(audio_dir)
        return audio, sample_rate
    
    def scale_audio(self, audio: Tensor, scaler: Callable) -> Tensor:
        scaled = scaler(audio)
        return scaled
    
    def highpass_filter(self, audio: Tensor, sample_rate: int, cutoff: int) -> Tensor:
        filtered_audio = torchaudio.functional.highpass_biquad(audio, sample_rate, cutoff)
        return filtered_audio

    def find_peaks(self, audio: Tensor, threshold: float) -> Tensor:
        peaks, _ = find_peaks(audio.numpy()[0], height=threshold)
        return peaks
    
    def trim_audio(self, audio: Tensor, peaks: array, n_peak: int) -> Tensor:
        trimmed_audio = audio[:, peaks[n_peak]:]
        return trimmed_audio
    
    def transform(self, audio_dir: str, scaler: Callable, cutoff: int, threshold: float, n_peak: int) -> Tensor:
        audio, sample_rate = self.load_audio(audio_dir)
        audio = self.scale_audio(audio, scaler)
        audio = self.highpass_filter(audio, sample_rate, cutoff)
        peaks = self.find_peaks(audio, threshold)
        trimmed_audio = self.trim_audio(audio, peaks, n_peak)
        return trimmed_audio