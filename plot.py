import numpy as np
import matplotlib.pyplot as plt


def plot_waveform_and_specgram(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    # Creamos dos filas de subplots, una para waveform y otra para specgram
    figure, axes = plt.subplots(2 * num_channels, 1, figsize=(10, 7 * num_channels))

    # Si tenemos un solo canal, axes será un array 1D,
    # lo convertimos en un array de objetos axes para mantener la consistencia
    if num_channels == 1:
        axes = np.array([axes[0], axes[1]])

    for c in range(num_channels):
        # Waveform para el canal c
        ax_waveform = axes[2 * c] if num_channels > 1 else axes[0]
        ax_waveform.plot(time_axis, waveform[c], linewidth=1)
        ax_waveform.grid(True)
        ax_waveform.set_title(
            f"Channel {c+1} Waveform" if num_channels > 1 else "Waveform"
        )
        if c < num_channels - 1:  # Solo añadimos la etiqueta al último subplot
            ax_waveform.set_xlabel("")

        # Spectrogram para el canal c
        ax_specgram = axes[2 * c + 1] if num_channels > 1 else axes[1]
        # Añadimos un piso de ruido antes de calcular el espectrograma
        piso_ruido = 1e-10
        Z = np.abs(np.fft.fft(waveform[c])) + piso_ruido
        Pxx, freqs, bins, im = ax_specgram.specgram(Z, Fs=sample_rate, scale="dB")
        ax_specgram.set_title(
            f"Channel {c+1} Spectrogram" if num_channels > 1 else "Spectrogram"
        )

    plt.tight_layout()
    plt.show()
