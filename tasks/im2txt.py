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
from PIL import Image
from io import BytesIO


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
                 folder: str,
                 image_size=224,
                 resize_ratio=0.75,
                 transform=None):
        """
        im2txt task dataset.

        Args:
            folder (str): Folder containing images and text
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize.
        """
        super().__init__()
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.resize_ratio = resize_ratio
        self.image_transform = transform

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
        image_file = self.image_files[key]

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

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return image_tensor, description


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_datadir,
                 dev_datadir,
                 batch_size=64,
                 image_size=224,
                 resize_ratio=0.75,
                 web_dataset=False,
                 wds_keys='img,cap',
                 world_size=1,
                 dataset_size=[int(1e9)],
                 nworkers=0):
        super().__init__()
        self.train_datadir = train_datadir
        self.dev_datadir = dev_datadir
        self.batch_size = batch_size
        self.image_size = image_size
        self.resize_ratio = resize_ratio
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

        self.transform_train = T.Compose([
            T.Lambda(self.fix_img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.transform_val = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.Lambda(self.fix_img),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def imagetransform(self, b):
        return Image.open(BytesIO(b))

    def decode(self, s):
        s = s.decode('utf-8')
        s = s.split('\n')
        s = list(filter(lambda t: len(t) > 0, s))
        return choice(s)

    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def setup(self, stage=None):
        if self.web_dataset:
            DATASET_TRAIN = web_dataset_helper(self.train_datadir)
            DATASET_VAL = web_dataset_helper(self.dev_datadir)

            myimg, mycap = (self.wds_keys.split(',')[0], self.wds_keys.split(',')[1])
            train_image_text_mapping = {
                            myimg: self.imagetransform,
                            mycap: self.decode
                        }
            train_image_mapping = {
                            myimg: self.transform_train
                        }
            val_image_text_mapping = {
                            myimg: self.imagetransform,
                            mycap: self.decode
                        }
            val_image_mapping = {
                            myimg: self.transform_val
                        }

            self.train = (
                wds.WebDataset(DATASET_TRAIN)
                .map_dict(**train_image_text_mapping)     
                .map_dict(**train_image_mapping)
                .to_tuple(myimg, mycap)
                .batched(self.batch_size, partial=False)                 
                )   
            self.valid = (
                wds.WebDataset(DATASET_VAL)                 
                .map_dict(**val_image_text_mapping)     
                .map_dict(**val_image_mapping)
                .to_tuple(myimg, mycap)
                .batched(self.batch_size, partial=False)                   
                )  

        else:
            self.train = Dataset(folder=self.train_datadir,
                                 transform=self.transform_train,
                                 image_size=self.image_size,
                                 resize_ratio=self.resize_ratio)
            self.valid = Dataset(folder=self.dev_datadir,
                                 transform=self.transform_val,
                                 image_size=self.image_size,
                                 resize_ratio=self.resize_ratio)

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
                is_image=True,
                return_images=False):
    """im2txt table."""
    table = [' || '] * len(x)
    x = perceiver.encode_image(x).float()
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