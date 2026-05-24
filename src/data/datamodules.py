from pathlib import Path

import pytorch_lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import HymenopteraDataset
from .transforms import get_train_transform, get_val_transform


class HymenopteraDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        preprocessed_data_dir = Path(cfg.dataset.preprocessed_dir)

        self.train_df_path = preprocessed_data_dir / cfg.dataset.df.train
        self.val_df_path = preprocessed_data_dir / cfg.dataset.df.val
        self.test_df_path = preprocessed_data_dir / cfg.dataset.df.test

        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.pin_memory = cfg.dataset.pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.classes = cfg.dataset.classes

    def setup(self, stage: str | None = None) -> None:
        self.train_transform = get_train_transform(self.cfg.dataset.transforms)
        self.val_transform = get_val_transform(self.cfg.dataset.transforms)

        self.train_dataset = HymenopteraDataset(
            self.train_df_path, transform=self.train_transform
        )
        self.val_dataset = HymenopteraDataset(
            self.val_df_path, transform=self.val_transform
        )
        self.test_dataset = HymenopteraDataset(
            self.test_df_path, transform=self.val_transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader | None:
        if not self.test_dataset:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def get_class_names(self) -> dict[str, int]:
        return self.classes
