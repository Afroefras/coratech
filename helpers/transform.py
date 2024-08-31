import torchaudio
import numpy as np
from torch import Tensor
import torchaudio.functional as F
from scipy.io.wavfile import write
from typing import Tuple, Callable, List
from torchaudio.transforms import Resample
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, decimate


def standard_scale(x: Tensor) -> Tensor:
    """
    Scales the input tensor to have zero mean and unit variance.
    """
    x_mean = x.mean()
    x_std = x.std() + 1e-10

    scaled = x.clone()
    scaled -= x_mean
    scaled /= x_std

    return scaled


def min_max_scale(x: Tensor) -> Tensor:
    """
    Scales the input tensor to the range [0, 1].
    """
    x_min = x.min()
    x_max = x.max()

    scaled = x.clone()
    scaled -= x_min
    scaled /= x_max - x_min

    return scaled


def generate_synthetic_wave(
    frequency: int,
    secs_duration: int,
    sample_rate: int = 4000,
    pre_silence_duration: int = 0,
    post_silence_duration: int = 0,
) -> Tuple[np.ndarray, int]:

    pre_silence = np.zeros(int(sample_rate * pre_silence_duration))
    post_silence = np.zeros(int(sample_rate * post_silence_duration))

    t = np.linspace(0, secs_duration, int(sample_rate * secs_duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    combined_wave = np.concatenate([pre_silence, wave, post_silence])

    return combined_wave, sample_rate


def save_wave_to_wav(
    wave: np.ndarray, sample_rate: int, filename: str, volume: float = 1.0
) -> None:
    wave = wave * volume
    wave = np.clip(wave, -1.0, 1.0)
    wave = np.int16(wave * 32767)
    write(filename, sample_rate, wave)


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


class TrimAfterTrigger:
    def __init__(self) -> None:
        pass

    def load_audio(self, audio_dir: str) -> Tuple[Tensor, int]:
        """
        Loads an audio file from the given directory.
        """
        audio, sample_rate = torchaudio.load(audio_dir)
        return audio, sample_rate

    def scale_audio(self, audio: Tensor, scaler: Callable) -> Tensor:
        """
        Scales the audio tensor using the given scaler function.
        """
        scaled = scaler(audio)
        return scaled

    def filter_freq(self, audio: Tensor, sample_rate: int, freq: int) -> Tensor:
        low_freq = freq - 1
        high_freq = freq + 1
        filtered = apply_bandpass_filter(audio, sample_rate, low_freq, high_freq)
        return filtered

    def audio_to_abs(self, audio: Tensor) -> Tensor:
        """
        Converts the audio tensor to its absolute value.
        """
        abs_audio = audio.abs()
        return abs_audio

    def downsample_audio(self, audio: Tensor, downsample_factor: int) -> np.array:
        """
        Downsamples the audio tensor.
        """
        downsampled = decimate(audio, downsample_factor)
        return downsampled

    def smooth_signal(self, audio: np.array, sigma: int) -> np.array:
        """
        Smoothes the audio np.array using a Gaussian filter.
        """
        smoothed = gaussian_filter1d(audio, sigma=sigma)
        return smoothed

    def signal_diff(self, signal: Tensor) -> Tensor:
        curve_diff = np.diff(signal.squeeze())
        return Tensor(curve_diff)

    def find_real_peaks(
        self, signal: Tensor, height: float, prominence: float, upsample_factor: int
    ) -> np.array:
        peaks, _ = find_peaks(signal, height=height, prominence=prominence)
        real_peaks = peaks * upsample_factor
        return real_peaks

    def split_signal(
        self,
        raw_audio: Tensor,
        peaks: np.array
    ) -> List[Tuple[float, Tensor]]:
        
        split_points = np.concatenate(([0], peaks, [raw_audio.shape[-1]]))
        audio = raw_audio.squeeze()
        segments = [
            audio[split_points[i]:split_points[i + 1]]
            for i in range(len(split_points) - 1)
        ]

        return segments

    def transform(
        self,
        audio_dir: str,
        synthetic_freq: int,
        downsample_factor: int,
        sigma_smooth: int,
        peaks_height: float,
        peaks_prominence: float,
        sample_rate_target: int=None,
    ) -> Tuple[List[Tensor], int]:
        
        audio, sample_rate = self.load_audio(str(audio_dir))

        if sample_rate_target is not None:
            if sample_rate != sample_rate_target:
                resampler = Resample(orig_freq=sample_rate, new_freq=sample_rate_target)
                audio = resampler(audio)
                sample_rate = sample_rate_target

        filtered = self.filter_freq(audio, sample_rate, synthetic_freq)
        
        abs_filtered = self.audio_to_abs(filtered)
        downsampled = self.downsample_audio(abs_filtered, downsample_factor)
        smoothed = self.smooth_signal(downsampled, sigma_smooth)

        curve_diff = self.signal_diff(smoothed)
        abs_diff = self.audio_to_abs(curve_diff)
        scaled_diff = self.scale_audio(abs_diff, scaler=min_max_scale)

        peaks = self.find_real_peaks(
            scaled_diff, peaks_height, peaks_prominence, downsample_factor
        )

        segments = self.split_signal(audio, peaks)

        return segments, sample_rate

    def trim_to_min_length(
        self,
        mobile_audio: Tensor,
        mobile_sample_rate: int,
        digital_audio: Tensor,
        digital_sample_rate: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Trims the mobile and digital audio tensors to the minimum duration
        between the two. This ensures both audio samples have the same length
        for subsequent processing.
        """
        mobile_seconds = mobile_audio.size(1) / mobile_sample_rate
        digital_seconds = digital_audio.size(1) / digital_sample_rate

        min_seconds = int(min(mobile_seconds, digital_seconds))

        mobile_audio = mobile_audio[:, : min_seconds * mobile_sample_rate]
        digital_audio = digital_audio[:, : min_seconds * digital_sample_rate]

        return mobile_audio, digital_audio

    def align_audios(
        self, mobile_dir: str, digital_dir: str
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Aligns the audio from a mobile recording and a digital stethoscope
        by trimming both to the minimum duration and ensuring the sample rate
        is consistent for both.
        """
        mobile_audio, mobile_sample_rate = self.transform(mobile_dir)
        digital_audio, digital_sample_rate = self.transform(digital_dir)

        mobile_audio, digital_audio = self.trim_to_min_length(
            mobile_audio, mobile_sample_rate, digital_audio, digital_sample_rate
        )
        return mobile_audio, mobile_sample_rate, digital_audio, digital_sample_rate
