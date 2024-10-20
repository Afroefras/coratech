import torch
import torchaudio
import numpy as np
from pywt import cwt
from torch import no_grad
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.signal import spectrogram
from helpers.audio_utils import get_positive_freq_and_magn


def plot_wavelet_spectrogram(
    audio: np.array,
    sample_rate: int,
    downsample_factor: int = 50,
    wavelet: str = "cmor1.5-1.0",
) -> None:
    """
    Plots the waveform and wavelet spectrogram of an audio signal.
    """

    audio = audio.squeeze()

    audio_downsampled = decimate(audio, downsample_factor)
    sample_rate_downsampled = sample_rate // downsample_factor

    widths = np.arange(1, 128)
    cwt_matrix, freqs = cwt(
        audio_downsampled, widths, wavelet, sampling_period=1 / sample_rate_downsampled
    )

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    time = np.arange(audio.shape[-1]) / sample_rate
    ax[0].plot(time, audio)
    ax[0].set_title("Forma de onda")
    ax[0].set_ylabel("Amplitud")

    im = ax[1].imshow(
        np.abs(cwt_matrix),
        extent=[0, len(audio) / sample_rate, 1, 128],
        cmap="Blues",
        aspect="auto",
    )
    ax[1].set_title("Espectrograma wavelet")
    ax[1].set_ylabel("Escala")
    ax[1].set_xlabel("Tiempo (segundos)")

    fig.colorbar(im, ax=ax[1], orientation="horizontal", label="Magnitud")

    plt.tight_layout()
    plt.show()


def plot_spectrograms(audio: np.array, sample_rate: int) -> None:
    """
    Plots the waveform, Fourier spectrogram, and Mel spectrogram of an audio signal.
    """
    audio = audio.squeeze()

    # Crear la figura y subplots
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Forma de onda
    time = np.arange(audio.shape[-1]) / sample_rate
    ax[0].plot(time, audio)
    ax[0].set_title("Forma de onda")
    ax[0].set_ylabel("Amplitud")
    ax[0].set_xlabel("Tiempo (segundos)")

    # 2. Espectrograma de Fourier (STFT)
    f, t, Sxx = spectrogram(audio, fs=sample_rate, nperseg=1024, noverlap=512)
    im1 = ax[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
    ax[1].set_title("Espectrograma de Fourier (STFT)")
    ax[1].set_ylabel("Frecuencia (Hz)")
    ax[1].set_xlabel("Tiempo (segundos)")
    fig.colorbar(im1, ax=ax[1], orientation="horizontal", label="Intensidad (dB)")

    # 3. Espectrograma de Mel usando torchaudio
    waveform = torch.tensor(audio).unsqueeze(0)  # Añadir dimensión de batch
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=128
    )
    mel_spec = mel_spec_transform(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    im2 = ax[2].imshow(
        mel_spec_db.squeeze().numpy(), aspect="auto", origin="lower", cmap="viridis"
    )
    ax[2].set_title("Espectrograma de Mel")
    ax[2].set_ylabel("Mel bins")
    ax[2].set_xlabel("Tiempo (frames)")
    fig.colorbar(im2, ax=ax[2], orientation="horizontal", label="Intensidad (dB)")

    # Ajustar el layout
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
