import matplotlib.pyplot as plt
import torch


def plot_waveform_and_specgram(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    num_rows = num_channels * 2
    figure, axes = plt.subplots(num_rows, 1, figsize=(12, 8))

    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        ax_waveform = axes[c * 2]  # Posición par para waveform
        ax_waveform.plot(time_axis, waveform[c], linewidth=1)
        ax_waveform.grid(True)
        ax_waveform.set_title(
            f"Channel {c+1} Waveform" if num_channels > 1 else "Waveform"
        )

        ax_specgram = axes[c * 2 + 1]  # Posición impar para specgram
        ax_specgram.specgram(waveform[c], Fs=sample_rate)
        ax_specgram.set_title(
            f"Channel {c+1} Spectrogram" if num_channels > 1 else "Spectrogram"
        )

    plt.tight_layout()
    plt.show()
