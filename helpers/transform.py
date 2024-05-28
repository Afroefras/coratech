import torchaudio
from torch import Tensor, fft
from typing import Tuple, Callable
from numpy import array, percentile, diff
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


class TrimAfterClicker:
    def __init__(self) -> None:
        """Initializes the TrimAfterClicker class."""
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

    def get_frequencies(self, audio: Tensor, sample_rate: int) -> Tensor:
        """
        Calculates the frequencies of the audio tensor.
        """
        frequencies = fft.fftfreq(audio.size(1), d=1 / sample_rate)
        return frequencies

    def get_frequency_percentile(
        self, frequencies: Tensor, percentile_num: float
    ) -> float:
        """
        Calculates the frequency percentile of the audio tensor.
        """
        freq_percentile = percentile(frequencies, percentile_num)
        return freq_percentile

    def highpass_filter(self, audio: Tensor, sample_rate: int, cutoff: int) -> Tensor:
        """
        Applies a highpass filter to the audio tensor.
        """
        filtered_audio = torchaudio.functional.highpass_biquad(
            audio, sample_rate, cutoff
        )
        return filtered_audio
    
    def audio_to_abs(self, audio: Tensor) -> Tensor:
        """
        Converts the audio tensor to its absolute value.
        """
        abs_audio = audio.abs()
        return abs_audio
    
    def downsample_audio(self, audio: Tensor, downsample_factor: int) -> Tensor:
        """
        Downsamples the audio tensor.
        """
        downsampled = decimate(audio, downsample_factor)
        return downsampled
    
    def smooth_signal(self, audio: Tensor, sigma: float) -> Tensor:
        """
        Smoothes the audio tensor.
        """
        smoothed = gaussian_filter1d(audio, sigma=sigma)
        return smoothed

    def find_peaks(self, audio: Tensor, prominence: float) -> array:
        """
        Finds the peaks in the audio tensor.
        """
        peaks, _ = find_peaks(audio.numpy()[0], prominence=prominence)
        return peaks
    
    def get_peaks_distances(self, peaks: array) -> array:
        """
        Calculates the distances between the peaks.
        """
        peaks_distances = diff(peaks)
        return peaks_distances
    
    def get_nth_peak(self, peaks: array, peaks_distances: array, distance_threshold: int, nth_peak: int) -> int:
        """
        Returns the index of the nth peak.
        """
        peaks_copy = peaks[1:].copy()
        peaks_mask = peaks_distances < distance_threshold
        peaks_under_threshold = peaks_copy[peaks_mask]
        peak = peaks_under_threshold[nth_peak]
        return peak
    
    def get_upsampled_peak(self, peak: int, upsample_factor: int) -> int:
        """
        Returns the upsampled peak.
        """
        upsampled_peak = peak * upsample_factor
        return upsampled_peak

    def trim_audio(
        self, audio: Tensor, sample_rate: int, peak: int
    ) -> Tensor:
        """
        Trims the audio tensor to the given peak.
        """
        begin_at = peak + sample_rate // 20
        trimmed = audio[:, begin_at:]
        return trimmed

    def transform(
        self,
        audio_dir: str,
    ) -> Tensor:
        """
        Transforms the audio tensor using the following steps:

        1. Load the audio file.
        2. Scale the audio tensor.
        3. Calculate the frequency percentile
        3. Apply a highpass filter given the previous percentile.
        4. Find the peaks in the audio tensor.
        5. Trim the audio tensor to the given peak.
        6. Scale the trimmed audio tensor to the range [0, 1].

        Args:
            audio_dir: The directory of the audio file.

        Returns:
            The transformed audio tensor.
        """
        audio, sample_rate = self.load_audio(audio_dir)
        scaled_audio = self.scale_audio(audio, standard_scale)

        frequencies = self.get_frequencies(scaled_audio, sample_rate)
        cutoff = self.get_frequency_percentile(frequencies, 99)
        filtered_audio = self.highpass_filter(scaled_audio, sample_rate, cutoff)

        peak_threshold = percentile(filtered_audio, 99.99)
        peaks = self.find_peaks(filtered_audio, peak_threshold)

        trimmed = self.trim_audio(audio, sample_rate, peaks, n_peak=-1)
        final_audio = min_max_scale(trimmed)

        return final_audio, sample_rate

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
