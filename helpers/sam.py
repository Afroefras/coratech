import numpy as np
from torch import Tensor
from typing import Tuple
import helpers.audio_utils as AU
from helpers.find_trigger import FindTrigger
from scipy.signal import find_peaks, decimate


class StudentAuscultationManikin(FindTrigger):
    def __init__(self) -> None:
        super().__init__()

    def find_heartbeats(
        self,
        audio: Tensor,
        downsample_factor: int,
        peaks_height: float,
        peaks_prominence: float,
    ) -> np.array:

        abs_audio = audio.abs()

        downsampled = decimate(abs_audio, downsample_factor)
        downsampled = Tensor(downsampled.copy())

        scaled = AU.min_max_scale(downsampled)
        peaks, _ = find_peaks(
            scaled.squeeze(), height=peaks_height, prominence=peaks_prominence
        )

        return peaks * downsample_factor

    def trim_at_first_heartbeat(
        self,
        audio: Tensor,
        sample_rate: int,
        snippet_secs: int,
        downsample_factor: int,
        peaks_height: float,
        peaks_prominence: float,
    ) -> int:

        snippet = AU.trim_audio(audio, sample_rate, end_at=snippet_secs)
        heartbeats = self.find_heartbeats(
            snippet, downsample_factor, peaks_height, peaks_prominence
        )

        heartbeat = heartbeats[0] / sample_rate

        trim = AU.trim_audio(audio, sample_rate, start_at=heartbeat)
        return trim

    def match_heartbeats(
        self,
        mobile_audio: Tensor,
        stethos_audio: Tensor,
        sample_rate: int,
        snippet_secs: int,
        downsample_factor: int,
        peaks_height: float,
        peaks_prominence: float,
    ) -> Tuple[Tensor, Tensor]:

        mobile_trim = self.trim_at_first_heartbeat(
            mobile_audio,
            sample_rate,
            snippet_secs,
            downsample_factor,
            peaks_height,
            peaks_prominence,
        )

        stethos_trim = self.trim_at_first_heartbeat(
            stethos_audio,
            sample_rate,
            snippet_secs,
            downsample_factor,
            peaks_height,
            peaks_prominence,
        )

        mobile, stethos = self.set_min_length(mobile_trim, stethos_trim)

        mobile_scaled = AU.min_max_scale(mobile)
        stethos_scaled = AU.min_max_scale(stethos)

        return mobile_scaled, stethos_scaled
