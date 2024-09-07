import numpy as np
from typing import Tuple
from scipy.io.wavfile import write


def generate_synthetic_wave(
    frequency: int,
    secs_duration: int,
    sample_rate: int = 4000,
    pre_silence_duration: int = 0,
    post_silence_duration: int = 0,
) -> Tuple[np.ndarray, int]:

    pre_silence = np.zeros(int(sample_rate * pre_silence_duration))
    post_silence = np.zeros(int(sample_rate * post_silence_duration))

    t = np.linspace(0, secs_duration, int(sample_rate * secs_duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    combined_wave = np.concatenate([pre_silence, wave, post_silence])

    return combined_wave, sample_rate


def save_wave_to_wav(
    wave: np.ndarray, sample_rate: int, filename: str, volume: float = 1.0
) -> None:
    wave = wave * volume
    bit_limit = 2**15 - 1

    wave = np.int16(wave * bit_limit)
    wave = np.clip(wave, -bit_limit, bit_limit)

    filename += ".wav"
    write(filename, sample_rate, wave)
    print(f"File '{filename}' was saved succesfully!")
