import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from helpers.audio_utils import standard_scale
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CoraTechDataset(Dataset):
    def __init__(self, base_dir: Path, chunk_secs: float):
        self.base_dir = base_dir
        self._load_data(chunk_secs)

    def __len__(self) -> int:
        return len(self.data)

    def make_heartbeats_chunks(
        self, audio: torch.Tensor, sample_rate: int, chunk_secs: float
    ) -> List[torch.Tensor]:
        chunk_size = int(sample_rate * chunk_secs)
        chunks = torch.split(audio, chunk_size, dim=-1)
        chunks = list(chunks)

        last_chunk_pad_size = chunk_size - chunks[-1].shape[-1]
        if last_chunk_pad_size > 0:
            chunks[-1] = standard_scale(chunks[-1])
            chunks[-1] = nn.functional.pad(chunks[-1], (0, last_chunk_pad_size))

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

        mobile = standard_scale(mobile)
        stethos = standard_scale(stethos)

        return mobile, stethos


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
