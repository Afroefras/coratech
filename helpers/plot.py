import numpy as np
from pywt import cwt
from torch import Tensor
import matplotlib.pyplot as plt

def plot_wavelet_spectrogram(audio_tensor: Tensor, sample_rate: int, wavelet: str='cmor1.5-1.0'):
    """
    Plots the waveform and wavelet spectrogram of an audio signal.

    Args:
    audio_tensor (Tensor): The audio signal as a PyTorch tensor of shape (1, n) or (n,).
    sample_rate (int): The sample rate of the audio signal.
    wavelet (str): The wavelet type to use for the continuous wavelet transform (CWT). Default is 'cmor1.5-1.0'.

    Returns:
    None
    """
    
    # Ensure the audio tensor is a numpy array
    audio = audio_tensor.squeeze().numpy()
    
    # Compute the continuous wavelet transform (CWT)
    widths = np.arange(1, 128)
    cwt_matrix, freqs = cwt(audio, widths, wavelet, sampling_period=1/sample_rate)

    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot the waveform
    time = np.arange(audio.size) / sample_rate
    ax[0].plot(time, audio)
    ax[0].set_title('Waveform')
    ax[0].set_ylabel('Amplitude')
    
    # Plot the wavelet spectrogram
    im = ax[1].imshow(np.abs(cwt_matrix), extent=[0, len(audio) / sample_rate, 1, 128], cmap='Blues', aspect='auto')
    ax[1].set_title('Wavelet Spectrogram')
    ax[1].set_ylabel('Scale')
    ax[1].set_xlabel('Time [sec]')
    
    # Add a colorbar for the spectrogram
    fig.colorbar(im, ax=ax[1], orientation='horizontal', label='Magnitude')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()