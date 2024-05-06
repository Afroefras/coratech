from typing import List, Tuple
from pathlib import Path, PosixPath


def find_pair(
    mobile_audio_dir: PosixPath, digital_dir: Path, digital_ext: str
) -> Tuple[PosixPath, PosixPath]:
    mobile_audio_name = mobile_audio_dir.stem

    digital_audio_name = mobile_audio_name + digital_ext
    digital_audio_dir = digital_dir.joinpath(digital_audio_name)

    return mobile_audio_dir, digital_audio_dir


def load_pairs(
    mobile_dir: str, mobile_ext: str, digital_dir: str, digital_ext: str
) -> List[Tuple[PosixPath, PosixPath]]:
    mobile_dir = Path(mobile_dir)
    digital_dir = Path(digital_dir)

    audios = []
    for audio in mobile_dir.glob(f"*{mobile_ext}"):
        mobile, digital = find_pair(
            mobile_audio_dir=audio, digital_dir=digital_dir, digital_ext=digital_ext
        )

        audios.append((mobile, digital))

    return audios
