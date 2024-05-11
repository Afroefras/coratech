import torchaudio
from torch import Tensor, fft
from typing import Tuple, Callable
from numpy import array, percentile
from scipy.signal import find_peaks


def standard_scale(x: Tensor) -> Tensor:
    """Scales the input tensor to have zero mean and unit variance.

    Args:
        x: The input tensor.

    Returns:
        The scaled tensor.
    """

    x_mean = x.mean()
    x_std = x.std() + 1e-10

    scaled = x.clone()
    scaled -= x_mean
    scaled /= x_std

    return scaled


def min_max_scale(x: Tensor) -> Tensor:
    """Scales the input tensor to the range [0, 1].

    Args:
        x: The input tensor.

    Returns:
        The scaled tensor.
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

    def load_audio(self, audio_dir: str) -> Tuple[Tensor, Tensor]:
        """Loads an audio file from the given directory.

        Args:
            audio_dir: The directory of the audio file.

        Returns:
            A tuple containing the audio tensor and the sample rate.
        """

        audio, sample_rate = torchaudio.load(audio_dir)
        return audio, sample_rate

    def scale_audio(self, audio: Tensor, scaler: Callable) -> Tensor:
        """Scales the audio tensor using the given scaler function.

        Args:
            audio: The audio tensor.
            scaler: The scaler function.

        Returns:
            The scaled audio tensor.
        """

        scaled = scaler(audio)
        return scaled

    def get_frequencies(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Calculates the frequencies of the audio tensor.

        Args:
            audio: The audio tensor.
            sample_rate: The sample rate of the audio.

        Returns:
            A tensor containing the frequencies.
        """

        frequencies = fft.fftfreq(audio.size(1), d=1 / sample_rate)
        return frequencies

    def get_magnitudes(self, audio: Tensor) -> Tensor:
        """Calculates the magnitudes of the audio tensor.

        Args:
            audio: The audio tensor.

        Returns:
            A tensor containing the magnitudes.
        """

        magnitudes = fft.fft(audio, dim=1).abs()
        return magnitudes

    def half_magnitudes_and_frequencies(
        self, magnitudes: Tensor, frequencies: Tensor
    ) -> Tensor:
        """Calculates the half magnitudes and frequencies of the
            audio tensor because FFT is symmetrical.

        Args:
            magnitudes: The magnitudes of the audio tensor.
            frequencies: The frequencies of the audio tensor.

        Returns:
            A tuple containing the half magnitudes and frequencies.
        """

        half_magnitudes = magnitudes[:, : magnitudes.size(1) // 2]
        half_frequencies = frequencies[: magnitudes.size(1) // 2]
        return half_magnitudes, half_frequencies

    def sort_frequencies(
        self, half_magnitudes: Tensor, half_frequencies: Tensor
    ) -> Tensor:
        """Sorts the half frequencies of the audio tensor.

        Args:
            half_magnitudes: The half magnitudes of the audio tensor.
            half_frequencies: The half frequencies of the audio tensor.

        Returns:
            A tensor containing the sorted half frequencies.
        """

        sorted_magnitudes = half_magnitudes.argsort(descending=True)
        sorted_frequencies = half_frequencies[sorted_magnitudes]
        return sorted_frequencies

    def get_frequency_percentile(
        self, audio: Tensor, sample_rate: int, percentile_num: float
    ) -> float:
        """Calculates the frequency percentile of the audio tensor.

        Args:
            audio: The audio tensor.
            sample_rate: The sample rate of the audio.
            percentile_num: The percentile to calculate.

        Returns:
            The float frequency percentile.
        """

        frequencies = self.get_frequencies(audio, sample_rate)
        magnitudes = self.get_magnitudes(audio)

        half_magnitudes, half_frequencies = self.half_magnitudes_and_frequencies(
            magnitudes, frequencies
        )
        sorted_frequencies = self.sort_frequencies(half_magnitudes, half_frequencies)

        freq_percentile = percentile(sorted_frequencies, percentile_num)
        return freq_percentile

    def highpass_filter(self, audio: Tensor, sample_rate: int, cutoff: int) -> Tensor:
        """Applies a highpass filter to the audio tensor.

        Args:
            audio: The audio tensor.
            sample_rate: The sample rate of the audio.
            cutoff: The cutoff frequency of the filter.

        Returns:
            The filtered audio tensor.
        """

        filtered_audio = torchaudio.functional.highpass_biquad(
            audio, sample_rate, cutoff
        )
        return filtered_audio

    def find_peaks(self, audio: Tensor, threshold: float) -> Tensor:
        """Finds the peaks in the audio tensor.

        Args:
            audio: The audio tensor.
            threshold: The threshold for peak detection.

        Returns:
            A tensor containing the indices of the peaks.
        """

        peaks, _ = find_peaks(audio.numpy()[0], height=threshold)
        return peaks

    def trim_audio(
        self, audio: Tensor, sample_rate: int, peaks: array, n_peak: int
    ) -> Tensor:
        """Trims the audio tensor to the given peak.

        Args:
            audio: The audio tensor.
            sample_rate: The sample rate of the audio.
            peaks: The indices of the peaks.
            n_peak: The index of the peak to trim to.

        Returns:
            The trimmed audio tensor.
        """

        begin_at = peaks[n_peak] + sample_rate // 20
        trimmed = audio[:, begin_at:]
        return trimmed

    def transform(
        self,
        audio_dir: str,
    ) -> Tensor:
        """Transforms the audio tensor using the following steps:

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

        cutoff = self.get_frequency_percentile(scaled_audio, sample_rate, 90)
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
        
        mobile_seconds = mobile_audio.size(1) / mobile_sample_rate
        digital_seconds = digital_audio.size(1) / digital_sample_rate

        min_seconds = int(min(mobile_seconds, digital_seconds))

        mobile_audio = mobile_audio[:, :min_seconds * mobile_sample_rate]
        digital_audio = digital_audio[:, :min_seconds * digital_sample_rate]

        return mobile_audio, digital_audio

    def align_audios(self, mobile_dir: str, digital_dir: str) -> Tuple[Tensor, Tensor]:
        mobile_audio, mobile_sample_rate = self.transform(mobile_dir)
        digital_audio, digital_sample_rate = self.transform(digital_dir)

        mobile_audio, digital_audio = self.trim_to_min_length(
            mobile_audio, mobile_sample_rate, digital_audio, digital_sample_rate
        )
        return (mobile_audio, mobile_sample_rate), (digital_audio, digital_sample_rate)
