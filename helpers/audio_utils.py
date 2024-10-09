import numpy as np
from torch import Tensor
from typing import Tuple
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.signal import butter, lfilter

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


def get_positive_freq_and_magn(
    audio: np.array, sample_rate: int
) -> Tuple[np.array, np.array]:
    audio = audio.squeeze()

    fft_result = np.fft.fft(audio)
    fft_magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

    positive_frequencies = frequencies[frequencies >= 0]
    positive_fft_magnitude = fft_magnitude[: len(positive_frequencies)]

    return positive_frequencies, positive_fft_magnitude


def apply_bandpass_filter(
    waveform: Tensor, sample_rate: int, low_freq: int, high_freq: int
) -> Tensor:

    central_freq = (low_freq + high_freq) / 2
    bandwidth = high_freq - low_freq
    Q = central_freq / bandwidth

    filtered_waveform = F.bandpass_biquad(waveform, sample_rate, central_freq, Q)
    return filtered_waveform


def apply_lowpass_filter(
    waveform: Tensor, sample_rate: int, cutoff_freq: int, order: int
) -> Tensor:
    # Calcula los coeficientes del filtro Butterworth pasa bajas
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    # Aplica el filtro usando lfilter de scipy
    filtered_waveform = lfilter(b, a, waveform.numpy())
    
    # Convierte de nuevo a Tensor de PyTorch
    return Tensor(filtered_waveform)


def trim_audio(
    audio: Tensor, sample_rate: int, start_at: float = None, end_at: float = None
):
    if start_at is None:
        start_at = 0
    if end_at is None:
        end_at = audio.shape[-1] // sample_rate

    starts = int(start_at * sample_rate)
    ends = int(end_at * sample_rate)

    return audio[:, starts:ends]


def resample_audio(
    audio: Tensor, sample_rate: int, new_sample_rate: int
) -> Tuple[Tensor, int]:
    resampler = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    audio = resampler(audio)
    sample_rate = new_sample_rate
    return audio, sample_rate


def add_noise(audio: Tensor, noise: Tensor, noise_vol: float) -> Tensor:
    if audio.shape[-1] > noise.shape[-1]:
        repeat_factor = audio.shape[-1] // noise.shape[-1] + 1
        noise = noise.repeat(1, repeat_factor)

    noise = noise[:, : audio.shape[-1]]

    audio_noisy = noise * noise_vol + audio * (1 - noise_vol)
    return audio_noisy
