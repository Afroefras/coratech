import numpy as np
from pywt import cwt
from torch import Tensor
import matplotlib.pyplot as plt
from scipy.signal import decimate


def plot_wavelet_spectrogram(
    audio_tensor: Tensor,
    sample_rate: int,
    downsample_factor: int = 50,
    wavelet: str = "cmor1.5-1.0",
):
    """
    Plots the waveform and wavelet spectrogram of an audio signal.

    Args:
    audio_tensor (Tensor): The audio signal as a PyTorch tensor of shape (1, n) or (n,).
    sample_rate (int): The sample rate of the audio signal.
    downsample_factor (int): The factor by which to downsample the audio signal. Default is 50.
    wavelet (str): The wavelet type to use for the continuous wavelet transform (CWT).

    Returns:
    None
    """

    # Ensure the audio tensor is a numpy array
    audio = audio_tensor.squeeze().numpy()

    # Downsample the audio signal
    audio_downsampled = decimate(audio, downsample_factor)
    sample_rate_downsampled = sample_rate // downsample_factor

    # Compute the continuous wavelet transform (CWT) on the downsampled audio
    widths = np.arange(1, 128)
    cwt_matrix, freqs = cwt(
        audio_downsampled, widths, wavelet, sampling_period=1 / sample_rate_downsampled
    )

    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # Plot the waveform
    time = np.arange(audio.size) / sample_rate
    ax[0].plot(time, audio)
    ax[0].set_title("Waveform")
    ax[0].set_ylabel("Amplitude")

    # Plot the wavelet spectrogram
    im = ax[1].imshow(
        np.abs(cwt_matrix),
        extent=[0, len(audio) / sample_rate, 1, 128],
        cmap="Blues",
        aspect="auto",
    )
    ax[1].set_title("Wavelet Spectrogram")
    ax[1].set_ylabel("Scale")
    ax[1].set_xlabel("Time [sec]")

    # Add a colorbar for the spectrogram
    fig.colorbar(im, ax=ax[1], orientation="horizontal", label="Magnitude")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
