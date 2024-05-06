import numpy as np
import matplotlib.pyplot as plt


def plot_waveform_and_specgram(waveform, sample_rate):
    """Plots the waveform and spectrogram of the given audio signal.

    Args:
        waveform: The audio signal waveform.
        sample_rate: The sample rate of the audio signal.
    """

    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    # Create two rows of subplots, one for waveform and one for specgram
    figure, axes = plt.subplots(2 * num_channels, 1, figsize=(10, 7 * num_channels))

    # If we have only one channel, axes will be a 1D array,
    # convert it to an array of axes objects for consistency
    if num_channels == 1:
        axes = np.array([axes[0], axes[1]])

    for c in range(num_channels):
        # Waveform for channel c
        ax_waveform = axes[2 * c] if num_channels > 1 else axes[0]
        ax_waveform.plot(time_axis, waveform[c], linewidth=1)
        ax_waveform.grid(True)
        ax_waveform.set_title(
            f"Channel {c+1} Waveform" if num_channels > 1 else "Waveform"
        )
        if c < num_channels - 1:  # Only add label to last subplot
            ax_waveform.set_xlabel("")

        # Spectrogram for channel c
        ax_specgram = axes[2 * c + 1] if num_channels > 1 else axes[1]
        # Add noise floor before computing spectrogram
        noise_floor = 1e-10
        Z = np.abs(np.fft.fft(waveform[c])) + noise_floor
        Pxx, freqs, bins, im = ax_specgram.specgram(Z, Fs=sample_rate, scale="dB")
        ax_specgram.set_title(
            f"Channel {c+1} Spectrogram" if num_channels > 1 else "Spectrogram"
        )

    plt.tight_layout()
    plt.show()
