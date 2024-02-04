from typing import Tuple
from pathlib import Path, PosixPath


def find_pair(
    mobile_dir: Path, mobile_audio_name: str, digital_dir: Path, digital_ext: str
) -> Tuple[PosixPath, PosixPath]:
    mobile_audio_dir = mobile_dir.joinpath(mobile_audio_name)
    mobile_audio_name = mobile_audio_dir.stem

    digital_audio_name = mobile_audio_name + digital_ext
    digital_audio_dir = digital_dir.joinpath(digital_audio_name)

    return mobile_audio_dir, digital_audio_dir
