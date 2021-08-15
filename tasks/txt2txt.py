import collections
import torch
import PIL
import pytorch_lightning as pl
import numpy as np
import torchvision.transforms.functional as F

from pathlib import Path
from torchvision import transforms as T
from random import randint, choice
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder: str):
        """
        txt2txt task dataset.

        Args:
            folder (str): Folder containing text files.
        """
        super().__init__()
        path = Path(folder)
        
        text_files = [*path.glob('**/*.txt')]
        self.text_files = {text_file.stem: text_file for text_file in text_files}
        self.keys = list(self.text_files.keys())
        
    def __len__(self):
        return len(self.keys)
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        try:
            descriptions = text_file.read_text().split('\n')
        except UnicodeDecodeError:
            return self.skip_sample(ind)
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        
        return description, description


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_datadir,
                 dev_datadir,
                 batch_size=64,
                 nworkers=0):
        super().__init__()
        self.train_datadir = train_datadir
        self.dev_datadir = dev_datadir
        self.batch_size = batch_size
        self.nworkers = nworkers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = Dataset(folder=self.train_datadir)
            self.valid = Dataset(folder=self.dev_datadir)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nworkers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.nworkers,
            pin_memory=True)


def build_table(x,
                perceiver,
                tokenize,
                indices,
                indices_data,
                device,
                knn,
                y=None,
                ctx=None,
                is_image=False,
                return_images=False):
    """txt2txt table."""
    table = [' || '] * len(x)
    if is_image:
        x = perceiver.encode_image(x).float()
    else:
        x = tokenize(x, truncate=True).to(device)
        x = perceiver.encode_text(x).float()
    x /= x.norm(dim=-1, keepdim=True)
    for (index, index_data) in zip(indices, indices_data):
        top_ind = index.search(x.cpu().numpy(), knn)[1]
        for idx in range(len(x)):
            results = [index_data[i] for i in top_ind[idx]]
            for r in results:
                table[idx] += r + ' | '       
    table = [r[:-1] + '|| ' for r in table]
    if y:
        table = [table[idx] + y[idx] for idx in range(len(x))]
    if return_images:
        return table, x
    return table