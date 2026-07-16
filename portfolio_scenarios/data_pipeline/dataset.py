import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data._utils.collate import default_collate
from collections import OrderedDict
from typing import Any, Tuple


# A copy and paste from the VAE papers authors code. It allows us to access the data in three ways
# .data/.labels , ['data']/['labels'] , [0]/[1]
class DatasetOutput(OrderedDict):
    r"""Same dataset output as in pythae library, inspired from
    the ``ModelOutput`` class from hugginface transformers library.

    This works with our BaseDataset, which uses DatasetOutput as output
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


def collate_dataset_output(batch):
    """Collate function that treats the `DatasetOutput` class correctly."""
    if isinstance(batch[0], DatasetOutput):
        # `default_collate` returns a dict for older versions of PyTorch.
        return DatasetOutput(**default_collate(batch))
    else:
        return default_collate(batch)

class TCVAEDataset(Dataset):
    def __init__(self, prices: np.ndarray, conditions: np.ndarray, window_size: int = 26):
        self.window_size = window_size

        self.windows = sliding_window_view(prices, window_shape=window_size, axis=0)
        self.window = self.windows.transpose(0, 2, 1)  # (n_windows, window_size, n_assets)
        self.conditions = conditions
        self.n_windows = self.windows.shape[0]

        # Pre-compute full normalised tensors so the trainer can access
        # eval_dataset.data and eval_dataset.labels as direct attributes.
        normalized = self.window / self.window[:, 0:1, :]   # (n_windows, W, n_assets)
        self.data   = torch.tensor(normalized, dtype=torch.float32)
        self.labels = torch.tensor(
            self.conditions[:self.n_windows], dtype=torch.float32
        )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        return DatasetOutput(data=self.data[idx], labels=self.labels[idx])

