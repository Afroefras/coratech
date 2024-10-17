import numpy as np
from pywt import cwt
from torch import no_grad
import matplotlib.pyplot as plt
from scipy.signal import decimate
from helpers.audio_utils import get_positive_freq_and_magn


def plot_wavelet_spectrogram(
    audio: np.array,
    sample_rate: int,
    downsample_factor: int = 50,
    wavelet: str = "cmor1.5-1.0",
):
    """
    Plots the waveform and wavelet spectrogram of an audio signal.
    """

    audio = audio.squeeze()

    # Downsample the audio signal
    audio_downsampled = decimate(audio, downsample_factor)
    sample_rate_downsampled = sample_rate // downsample_factor

    # Compute the continuous wavelet transform (CWT) on the downsampled audio
    widths = np.arange(1, 128)
    cwt_matrix, freqs = cwt(
        audio_downsampled, widths, wavelet, sampling_period=1 / sample_rate_downsampled
    )

    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Plot the waveform
    time = np.arange(audio.shape[-1]) / sample_rate
    ax[0].plot(time, audio)
    ax[0].set_title("Forma de onda")
    ax[0].set_ylabel("Amplitud")

    # Plot the wavelet spectrogram
    im = ax[1].imshow(
        np.abs(cwt_matrix),
        extent=[0, len(audio) / sample_rate, 1, 128],
        cmap="Blues",
        aspect="auto",
    )
    ax[1].set_title("Espectrograma wavelet")
    ax[1].set_ylabel("Escala")
    ax[1].set_xlabel("Tiempo (segundos)")

    # Add a colorbar for the spectrogram
    fig.colorbar(im, ax=ax[1], orientation="horizontal", label="Magnitud")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_audio_fft(audio: np.array, sample_rate: int) -> None:
    freq, magn = get_positive_freq_and_magn(audio, sample_rate)

    plt.figure(figsize=(12, 3))
    plt.plot(freq, magn)
    plt.title("Distribución de Frecuencias")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()


def compare_audios(
    first_audio: np.array,
    first_title: str,
    second_audio: np.array,
    second_title: str,
    sample_rate: int,
) -> None:
    time_axis = np.linspace(0, len(second_audio) / sample_rate, num=len(second_audio))

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_axis, first_audio, label=first_title, alpha=0.7, color="skyblue")
    plt.plot(time_axis, second_audio, label=second_title, alpha=0.7)
    plt.ylabel("Amplitud")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_axis, first_audio, label=first_title, alpha=0.7, color="skyblue")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_axis, second_audio, label=second_title, alpha=0.7)
    plt.xlabel("Tiempo (segundos)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_model_result(trained_model, dataset, index):
    trained_model.eval()
    with no_grad():
        mobile, stethos = dataset[index]
        model_result = trained_model(mobile.unsqueeze(0))

        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        axs[0].plot(mobile.squeeze())
        axs[0].set_title("Celular")

        axs[1].plot(stethos.squeeze())
        axs[1].set_title("Estetoscopio")

        axs[2].plot(model_result.squeeze())
        axs[2].set_title("Modelo")

        plt.tight_layout()
        plt.show()

        return model_result
