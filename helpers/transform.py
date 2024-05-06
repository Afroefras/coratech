import torchaudio
from numpy import array
from torch import Tensor
from typing import Tuple, Callable
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


class AlignAudios:
    def __init__(self) -> None:
        """Initializes the AlignAudios class."""

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

        begin_at = peaks[n_peak] + sample_rate // 60
        trimmed = audio[:, begin_at:]
        return trimmed

    def transform(
        self,
        audio_dir: str,
        cutoff: int = 600,
        threshold: float = 0.7,
        n_peak: int = -1,
    ) -> Tensor:
        """Transforms the audio tensor using the following steps:

        1. Load the audio file.
        2. Scale the audio tensor.
        3. Apply a highpass filter to the audio tensor.
        4. Find the peaks in the audio tensor.
        5. Trim the audio tensor to the given peak.
        6. Scale the trimmed audio tensor to the range [0, 1].

        Args:
            audio_dir: The directory of the audio file.
            cutoff: The cutoff frequency of the highpass filter.
            threshold: The threshold for peak detection.
            n_peak: The index of the peak to trim to.

        Returns:
            The transformed audio tensor.
        """

        audio, sample_rate = self.load_audio(audio_dir)
        audio = self.scale_audio(audio, standard_scale)
        audio = self.highpass_filter(audio, sample_rate, cutoff)
        peaks = self.find_peaks(audio, threshold)
        trimmed = self.trim_audio(audio, sample_rate, peaks, n_peak)
        final_audio = min_max_scale(trimmed)
        return final_audio, sample_rate
