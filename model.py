import torch
import random
import torchaudio
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helpers.audio_utils import (
    standard_scale,
    add_noise,
    resample_audio,
    apply_lowpass_filter,
)


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

        mobile = mobile.squeeze(0)
        stethos = stethos.squeeze(0)
        return mobile, stethos


class AddHospitalNoise(object):
    def __init__(
        self,
        noise_dir: Path,
        noise_volume_range: Tuple[float, float],
        sample_rate_target: int,
    ) -> None:
        self.noise_files = list(noise_dir.glob("*.wav"))
        self.noise_volume_range = noise_volume_range
        self.sample_rate_target = sample_rate_target

    def load_noise(self) -> torch.Tensor:
        noise_path = random.choice(self.noise_files)

        noise, sample_rate = torchaudio.load(str(noise_path))
        noise, sample_rate = resample_audio(noise, sample_rate, self.sample_rate_target)
        noise = standard_scale(noise)
        return noise

    def get_random_noise_snippet(
        self, noise: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        if noise.shape[-1] <= n_samples:
            return noise

        start_idx = random.randint(0, noise.shape[-1] - n_samples)
        end_idx = start_idx + n_samples
        return noise[:, start_idx:end_idx]

    def __call__(self, sample) -> Dict[str, torch.Tensor]:
        mobile, stethos = sample["mobile"], sample["stethos"]

        hospital_noise = self.load_noise()
        noise_snippet = self.get_random_noise_snippet(hospital_noise, mobile.shape[-1])

        noise_volume = random.uniform(*self.noise_volume_range)
        noisy_mobile = add_noise(mobile, noise_snippet, noise_volume)

        return {"mobile": noisy_mobile, "stethos": stethos}


class LowpassFilter(object):
    def __init__(self, sample_rate: int, cutoff_freq: int, order: int):
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.order = order

    def __call__(self, sample):
        mobile, stethos = sample["mobile"], sample["stethos"]

        mobile_filtered = apply_lowpass_filter(
            mobile,
            sample_rate=self.sample_rate,
            cutoff_freq=self.cutoff_freq,
            order=self.order,
        )

        stethos_filtered = apply_lowpass_filter(
            stethos,
            sample_rate=self.sample_rate,
            cutoff_freq=self.cutoff_freq,
            order=self.order,
        )

        return {"mobile": mobile_filtered, "stethos": stethos_filtered}


class Normalize(object):
    def __call__(self, sample):
        mobile, stethos = sample["mobile"], sample["stethos"]

        mobile = standard_scale(mobile)
        stethos = standard_scale(stethos)

        return {"mobile": mobile, "stethos": stethos}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class CoraTechModel(LightningModule):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, input_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        output = self.fc3(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = nn.L1Loss()(predictions, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = nn.L1Loss()(predictions, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, patience=3),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
