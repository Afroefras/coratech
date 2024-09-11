import torch
import torchaudio
import numpy as np
from torch import Tensor
from pathlib import Path
from typing import Tuple, List
import torch.nn.functional as F
import helpers.audio_utils as AU
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, decimate


class FindTrigger:
    def __init__(self) -> None:
        pass

    def load_recordings(
        self, mobile_dir: str, stethos_dir: str
    ) -> Tuple[Tensor, Tensor, int]:

        stethos_audio, stethos_sample_rate = torchaudio.load(stethos_dir)
        mobile_audio, mobile_sample_rate = torchaudio.load(mobile_dir)

        mobile_audio, mobile_sample_rate = AU.resample_audio(
            mobile_audio, mobile_sample_rate, stethos_sample_rate
        )

        return mobile_audio, stethos_audio, stethos_sample_rate

    def set_min_length(self, mobile: Tensor, stethos: Tensor) -> Tuple[Tensor, Tensor]:
        min_length = min(mobile.shape[-1], stethos.shape[-1])

        mobile = mobile[:, :min_length].clone()
        stethos = stethos[:, :min_length].clone()

        return mobile, stethos

    def find_trigger_peaks(
        self,
        audio: Tensor,
        sample_rate: int,
        synthetic_freq: int,
        downsample_factor: int,
        sigma_smooth: int,
        peaks_height: float,
        peaks_prominence: float,
        snippet_secs: float = None,
    ) -> Tuple[List[Tensor], int]:

        snippet = AU.trim_audio(audio, sample_rate, end_at=snippet_secs)

        low_freq = synthetic_freq - 1
        high_freq = synthetic_freq + 1
        filtered = AU.apply_bandpass_filter(snippet, sample_rate, low_freq, high_freq)

        abs_filtered = filtered.abs()

        downsampled = decimate(abs_filtered, downsample_factor)

        smoothed = gaussian_filter1d(downsampled, sigma=sigma_smooth)

        curve_diff = np.diff(smoothed.squeeze())
        curve_diff = Tensor(curve_diff)

        abs_diff = curve_diff.abs()

        scaled_diff = AU.min_max_scale(abs_diff)

        peaks, _ = find_peaks(
            scaled_diff, height=peaks_height, prominence=peaks_prominence
        )

        return peaks * downsample_factor

    def trim_at_last_trigger(
        self,
        audio: Tensor,
        sample_rate: int,
        synthetic_freq: int,
        downsample_factor: int,
        sigma_smooth: int,
        peaks_height: float,
        peaks_prominence: float,
        snippet_secs: float,
    ) -> Tensor:

        triggers = self.find_trigger_peaks(
            audio,
            sample_rate,
            synthetic_freq,
            downsample_factor,
            sigma_smooth,
            peaks_height,
            peaks_prominence,
            snippet_secs,
        )
        last_trigger = triggers[-1] / sample_rate

        trim = AU.trim_audio(audio, sample_rate, start_at=last_trigger)
        return trim

    def match_last_trigger(
        self,
        mobile_audio: Tensor,
        stethos_audio: Tensor,
        sample_rate: int,
        synthetic_freq: int,
        downsample_factor: int,
        sigma_smooth: int,
        peaks_height: float,
        peaks_prominence: float,
        snippet_secs: float,
    ) -> Tuple[Tensor, Tensor]:

        mobile_trim = self.trim_at_last_trigger(
            mobile_audio,
            sample_rate,
            synthetic_freq,
            downsample_factor,
            sigma_smooth,
            peaks_height,
            peaks_prominence,
            snippet_secs,
        )
        stethos_trim = self.trim_at_last_trigger(
            stethos_audio,
            sample_rate,
            synthetic_freq,
            downsample_factor,
            sigma_smooth,
            peaks_height,
            peaks_prominence,
            snippet_secs,
        )

        mobile, stethos = self.set_min_length(mobile_trim, stethos_trim)

        mobile_scaled = AU.min_max_scale(mobile)
        stethos_scaled = AU.min_max_scale(stethos)

        return mobile_scaled, stethos_scaled

    def make_heartbeats_chunks(
        self, audio: Tensor, sample_rate: int, chunk_secs: float
    ) -> Tensor:
        chunk_size = int(sample_rate * chunk_secs)
        chunks = torch.split(audio, chunk_size, dim=-1)

        if chunks[-1].size(-1) < chunk_size:
            padding_size = chunk_size - chunks[-1].size(-1)
            chunks = list(chunks)
            chunks[-1] = F.pad(chunks[-1], (0, padding_size))

        return torch.stack(chunks, dim=0)

    def save_match(
        self,
        mobile: Tensor,
        stethos: Tensor,
        sample_rate: int,
        output_dir: str,
        filename: str,
        suffix: str = None,
    ) -> None:

        file_dir = Path(output_dir)
        base_filename = Path(filename).stem
        suffix = "" if suffix is None else f"_{suffix}"

        filename = f"{base_filename}{suffix}.pt"
        filepath = file_dir.joinpath(filename)

        torch.save((mobile, stethos, sample_rate), filepath)
        print(f"File '{filepath}' was saved succesfully!")
