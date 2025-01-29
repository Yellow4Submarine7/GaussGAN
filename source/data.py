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


class GaussianDataset(Dataset):

    def __init__(self, n_samples, mean1, cov1, mean2, cov2):


        # Generate 500 points from each distribution
        points1 = np.random.multivariate_normal(mean1, cov1, n_samples)
        points2 = np.random.multivariate_normal(mean2, cov2, n_samples)

        # Combine the points into a single dataset
        self.dataset = np.vstack((points1, points2))


    def __len__(self):
        """Returns the number of valid molecules in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset by index."""
        return self.dataset[idx]


    def save(self, filename):
        """Saves the dataset to a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {filename}.")

    @classmethod
    def load(cls, filename):
        """Loads the dataset from a pickle file."""
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {filename}.")
        return dataset

   



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
            #collate_fn=collate_fn,
        )

    def val_dataloader(self):
        # Returns the validation dataloader.
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            #collate_fn=collate_fn,
        )

    def test_dataloader(self):
        # Returns the testing dataloader.
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            #collate_fn=collate_fn,
        )
