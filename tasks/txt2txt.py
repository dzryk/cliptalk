import collections
import torch
import PIL
import pytorch_lightning as pl
import numpy as np
import torchvision.transforms.functional as F
import webdataset as wds

from pathlib import Path
from torchvision import transforms as T
from random import randint, choice
from torch.utils.data import DataLoader


def web_dataset_helper(path):
    """
    https://github.com/tgisaturday/dalle-lightning/blob/master/pl_dalle/loader.py
    """
    if Path(path).is_dir():
        DATASET = [str(p) for p in Path(path).glob("**/*") if ".tar" in str(p).lower()] # .name
        assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(path)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), path))
    elif ('http://' in path.lower()) | ('https://' in path.lower()):
        DATASET = f"pipe:curl -L -s {path} || true"
        print('Found {} http(s) link under given path!'.format(len(DATASET), path))
    elif 'gs://' in path.lower():
        DATASET = f"pipe:gsutil cat {path} || true"
        print('Found {} GCS link under given path!'.format(len(DATASET), path))
    elif '.tar' in path:
        DATASET = path
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(path))
    else:
        raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(path))
    return DATASET


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

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
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
                 web_dataset=False,
                 wds_keys='cap,',
                 world_size=1,
                 dataset_size=[int(1e9)],
                 nworkers=0):
        super().__init__()
        self.train_datadir = train_datadir
        self.dev_datadir = dev_datadir
        self.batch_size = batch_size
        self.web_dataset = web_dataset
        self.wds_keys = wds_keys
        self.world_size = world_size
        if len(dataset_size) == 1:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[0]
        else:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[1] 
        self.nworkers = nworkers

    def decode(self, s):
        s = s.decode('utf-8')
        s = s.split('\n')
        s = list(filter(lambda t: len(t) > 0, s))
        return choice(s)

    def setup(self, stage=None):
        if self.web_dataset:
            DATASET_TRAIN = web_dataset_helper(self.train_datadir)
            DATASET_VAL = web_dataset_helper(self.dev_datadir)

            mycap = self.wds_keys.split(',')[0]
            text_mapping = {
                            mycap: self.decode
                        }

            self.train = (
                wds.WebDataset(DATASET_TRAIN)
                .map_dict(**text_mapping)
                .to_tuple(mycap, mycap)
                .batched(self.batch_size, partial=False)                 
                )   
            self.valid = (
                wds.WebDataset(DATASET_VAL)                 
                .map_dict(**text_mapping)
                .to_tuple(mycap, mycap)
                .batched(self.batch_size, partial=False)                   
                )

        else:
            self.train = Dataset(folder=self.train_datadir)
            self.valid = Dataset(folder=self.dev_datadir)

    def train_dataloader(self):
        if self.web_dataset:
            dl = wds.WebLoader(self.train, batch_size=None, shuffle=False)
            number_of_batches = self.train_dataset_size // (self.batch_size * self.world_size)
            dl = dl.repeat(9999999999).slice(number_of_batches)
            dl.length = number_of_batches
            return dl
        else:
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.nworkers,
                pin_memory=True)

    def val_dataloader(self):
        if self.web_dataset:
            dl = wds.WebLoader(self.valid, batch_size=None, shuffle=False)
            number_of_batches = self.val_dataset_size // (self.batch_size * self.world_size)
            dl = dl.repeat(9999999999).slice(number_of_batches)
            dl.length = number_of_batches
            return dl
        else:
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
    if ctx:
        for idx in range(len(x)):
            table[idx] += ctx[idx] + ' || '
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