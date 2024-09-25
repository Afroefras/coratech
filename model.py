import torch
import random
import torchaudio
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helpers.audio_utils import standard_scale, add_noise


class CoraTechDataset(Dataset):
    def __init__(self, base_dir: Path, chunk_secs: float, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self._load_data(chunk_secs)

    def __len__(self) -> int:
        return len(self.data)

    def make_heartbeats_chunks(
        self, audio: torch.Tensor, sample_rate: int, chunk_secs: float
    ) -> List[torch.Tensor]:
        chunk_size = int(sample_rate * chunk_secs)
        chunks = torch.split(audio, chunk_size, dim=-1)
        chunks = list(chunks)

        if chunk_size > chunks[-1].shape[-1]:
            chunks.pop(-1)

        return chunks

    def _load_data(self, chunk_secs: float) -> None:
        self.data = []
        self.data_dir = list(self.base_dir.glob("*.pt"))

        for path in self.data_dir:
            mobile, stethos, sample_rate = torch.load(path)

            mobile_chunks = self.make_heartbeats_chunks(mobile, sample_rate, chunk_secs)
            stethos_chunks = self.make_heartbeats_chunks(
                stethos, sample_rate, chunk_secs
            )

            chunks = list(zip(mobile_chunks, stethos_chunks))
            self.data.extend(chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mobile, stethos = self.data[idx]

        if self.transform:
            sample = {"mobile": mobile, "stethos": stethos}
            sample = self.transform(sample)
            mobile, stethos = sample["mobile"], sample["stethos"]

        return mobile, stethos


class AddHospitalNoise(object):
    def __init__(self, noise_dir: Path, noise_volume_range=(0.05, 0.2)):
        self.noise_files = list(noise_dir.glob("*.wav"))
        self.noise_volume_range = noise_volume_range

    def get_random_noise_snippet(
        self, noise: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        if noise.shape[-1] <= n_samples:
            return noise

        start_idx = random.randint(0, noise.shape[-1] - n_samples)
        end_idx = start_idx + n_samples
        return noise[start_idx:end_idx]

    def __call__(self, sample):
        mobile, stethos = sample["mobile"], sample["stethos"]

        noise_path = random.choice(self.noise_files)
        hospital_noise = torchaudio.load(str(noise_path))
        noise_snippet = self.get_random_noise_snippet(hospital_noise, mobile.shape[-1])

        noise_volume = random.uniform(*self.noise_volume_range)
        noisy_mobile = add_noise(mobile, noise_snippet, noise_volume)

        return {"mobile": noisy_mobile, "stethos": stethos}


class Normalize(object):
    def __call__(self, sample):
        mobile, stethos = sample["mobile"], sample["stethos"]

        mobile = standard_scale(mobile)
        stethos = standard_scale(stethos)

        return {"mobile": mobile, "stethos": stethos}


class CoraTechModel(LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(64, input_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]
        output = self.fc(lstm_output)
        return output.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = nn.MSELoss()(predictions, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = nn.MSELoss()(predictions, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, patience=3),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
