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
    
    def trim_audio(self, audio: Tensor, sample_rate: int, peaks: array, n_peak: int) -> Tensor:
        begin_at = peaks[n_peak] + sample_rate // 60
        trimmed = audio[:, begin_at:]
        return trimmed
    
    def transform(self, audio_dir: str, cutoff: int=600, threshold: float=0.7, n_peak: int=-1) -> Tensor:
        audio, sample_rate = self.load_audio(audio_dir)
        audio = self.scale_audio(audio, standard_scale)
        audio = self.highpass_filter(audio, sample_rate, cutoff)
        peaks = self.find_peaks(audio, threshold)
        trimmed = self.trim_audio(audio, sample_rate, peaks, n_peak)
        final_audio = min_max_scale(trimmed)
        return final_audio, sample_rate