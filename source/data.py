import pickle
from typing import NamedTuple

import numpy as np
import torch
from lightning import LightningDataModule
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Data
from torchdrug.data import Molecule
from tqdm import tqdm

from .metrics import ALL_METRICS


class GaussianDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        batch_size=32,
        train_test_val_split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        # Save the batch size as a hyperparameter
        self.save_hyperparameters("batch_size")
        # Save the dataset
        self.dataset = dataset
        self.train_test_val_split = train_test_val_split
        # Initialize the train, validation, and test datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # This method should only run on 1 GPU/TPU in distributed settings,
        # thus we do not need to set anything related to the dataset itself here,
        # since it's done in setup() which is called on every GPU/TPU.
        pass

    def setup(self, stage=None):
        # Calculate split sizes based on the provided tuple ratios
        train_size = int(self.train_test_val_split[0] * len(self.dataset))
        val_size = int(self.train_test_val_split[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        # Returns the training dataloader.
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0 #collate_fn=collate_fn,
        )

    def val_dataloader(self):
        # Returns the validation dataloader.
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            #collate_fn=collate_fn,
        )

    def test_dataloader(self):
        # Returns the testing dataloader.
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            #collate_fn=collate_fn,
        )
