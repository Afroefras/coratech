import torch
from torch import Tensor
from pathlib import Path
from typing import List, Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset
from helpers.audio_utils import standard_scale

class CoraTechDataset(Dataset):
    def __init__(self, base_dir: Path, chunk_secs: float):
        self.base_dir = base_dir
        self._load_data(chunk_secs)

    def __len__(self) -> int:
        return len(self.data)

    def make_heartbeats_chunks(self, audio: torch.Tensor, sample_rate: int, chunk_secs: float) -> List[Tensor]:
        chunk_size = int(sample_rate * chunk_secs)
        chunks = torch.split(audio, chunk_size, dim=-1)
        chunks = list(chunks)

        last_chunk_pad_size = chunk_size - chunks[-1].shape[-1]
        if last_chunk_pad_size > 0:
            chunks[-1] = F.pad(chunks[-1], (0, last_chunk_pad_size))

        return chunks
    
    def _load_data(self, chunk_secs: float) -> None:
        self.data = []
        self.data_dir = list(self.base_dir.glob('*.pt'))

        for path in self.data_dir:
            mobile, stethos, sample_rate = torch.load(path)

            mobile_chunks = self.make_heartbeats_chunks(mobile, sample_rate, chunk_secs)
            stethos_chunks = self.make_heartbeats_chunks(stethos, sample_rate, chunk_secs)
            
            chunks = list(zip(mobile_chunks, stethos_chunks))
            self.data.extend(chunks)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        mobile, stethos = self.data[idx]

        mobile = standard_scale(mobile)
        stethos = standard_scale(stethos)
        
        return mobile, stethos
