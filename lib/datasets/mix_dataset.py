import numpy as np
import torch
from ..utils.builder import build_dataset
from ..utils.logger import logger
from termcolor import colored
from lib.utils.config import CN
import random

# class MixedWebDataset(wds.WebDataset):

#     def __init__(self, cfg_list, data_preset, transform, **kwargs) -> None:
#         super(wds.WebDataset, self).__init__()
#         datasets = [build_dataset(cfg, data_preset=data_preset, transform=transform, **kwargs) for cfg in cfg_list]
#         weights = self.ratios = np.array([cfg.MIX_RATIO for cfg in cfg_list])
#         weights = weights / weights.sum()  # normalize
#         self.append(wds.RandomMix(datasets, weights))


class MixDataset(torch.utils.data.Dataset):

    def __init__(self, cfg_list: list, data_preset: CN, transform: CN, length: int = None, **kwargs) -> None:

        self.datasets = [build_dataset(cfg, data_preset=data_preset, transform=transform, **kwargs) for cfg in cfg_list]
        self.ratios = np.array([cfg.MIX_RATIO for cfg in cfg_list])
        self.ratios = self.ratios / self.ratios.sum()  # normalize
        self.num_dataset = len(self.datasets)
        self.partitions = self.ratios.cumsum()
        self.len_dataset = [len(d) for d in self.datasets]

        assert self.partitions[-1] == 1.0, "Mixing ratios must sum to 1.0"

        self.total_length = sum(self.len_dataset)
        if length is not None:
            self.length = length
        else:
            self.length = max(self.len_dataset)

        logger.warning(f"MixDataset initialized Done! "
                       f"Including {len(self.datasets)} datasets and {self.length} working length")

        info = colored(" + ", 'blue', attrs=['bold']).\
            join([f"{self.ratios[i]} * {self.datasets[i].name}" for i in range(self.num_dataset)])
        logger.info(f"MixDataset: {info}")

    def __len__(self):
        return self.length

    def _common_keys(self):
        # Get the common keys of Dict [List]

        key_set = set(self.datasets[0][0].keys())  # First element of the first dataset

        for d in self.datasets[1:]:
            key_set = key_set & set(d[0].keys())

        return key_set

    def _data_filter(self, raw_data):
        common_keys = self._common_keys()
        targ_data = {each: raw_data[each] for each in common_keys}
        return targ_data

    def __getitem__(self, idx):
        """
        Index an element from the dataset.
        This is done by randomly choosing a dataset using the mixing percentages
        and then randomly choosing from the selected dataset.
        
        That is, the given idx becomes a dummy variable that won't matter
        
        Returns:
            Dict: Dictionary containing data and labels for the selected example
        """

        p = np.random.rand()
        for i in range(len(self.datasets)):  # N datasets
            if p <= self.partitions[i]:
                while True:
                    try:
                        targ_idx = random.randint(0, self.len_dataset[i] - 1)
                        return self._data_filter(self.datasets[i][targ_idx])  # Filter different keys for merging
                    except Exception as e:
                        logger.error(f"Error in MixDataset: {e}")
                        assert False
