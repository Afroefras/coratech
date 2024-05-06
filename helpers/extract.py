from typing import List, Tuple
from pathlib import Path, PosixPath


def find_pair(
    mobile_audio_dir: PosixPath, digital_dir: Path, digital_ext: str
) -> Tuple[PosixPath, PosixPath]:
    """Finds the corresponding digital audio file for the given mobile audio file.

    Args:
        mobile_audio_dir: The directory of the mobile audio file.
        digital_dir: The directory of the digital audio files.
        digital_ext: The extension of the digital audio files.

    Returns:
        A tuple containing the mobile audio file and the corresponding digital audio file.
    """

    mobile_audio_name = mobile_audio_dir.stem

    digital_audio_name = mobile_audio_name + digital_ext
    digital_audio_dir = digital_dir.joinpath(digital_audio_name)

    return mobile_audio_dir, digital_audio_dir


def load_pairs(
    mobile_dir: str, mobile_ext: str, digital_dir: str, digital_ext: str
) -> List[Tuple[PosixPath, PosixPath]]:
    """Loads pairs of mobile and digital audio files from the given directories.

    Args:
        mobile_dir: The directory of the mobile audio files.
        mobile_ext: The extension of the mobile audio files.
        digital_dir: The directory of the digital audio files.
        digital_ext: The extension of the digital audio files.

    Returns:
        A list of tuples containing the mobile audio file and the corresponding digital audio file.
    """

    mobile_dir = Path(mobile_dir)
    digital_dir = Path(digital_dir)

    audios = []
    for audio in mobile_dir.glob(f"*{mobile_ext}"):
        mobile, digital = find_pair(
            mobile_audio_dir=audio, digital_dir=digital_dir, digital_ext=digital_ext
        )

        audios.append((mobile, digital))

    return audios
