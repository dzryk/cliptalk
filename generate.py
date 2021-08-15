import os
import json
import glob
import torch
import numpy as np
import faiss
import PIL

from CLIP import clip
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

import model
import retrofit


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dl_collate_fn(batch):
    return torch.stack([row[0] for row in batch]), [row[1] for row in batch]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder: str,
                 image_size=224):
        super().__init__()
        path = Path(folder)

        image_files = sorted([
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ])

        self.image_files = {image_file.stem: image_file for image_file in image_files}
        self.keys = list(self.image_files.keys())
        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.Lambda(self.fix_img),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.keys)

    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

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
        image_file = self.image_files[key]
        cap_file = image_file.with_suffix('.cap')

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        return image_tensor, cap_file


def clip_rescoring(args, net, candidates, x):
    textemb = net.perceiver.encode_text(
        clip.tokenize(candidates).to(args.device)).float()
    textemb /= textemb.norm(dim=-1, keepdim=True)
    similarity = x @ textemb.T
    _, indices = similarity[0].topk(args.num_return_sequences)
    return [candidates[idx] for idx in indices[0]]


def caption_image(table, x, args, net):
    captions = []
    table = net.tokenizer.encode(table[0], return_tensors='pt').to(args.device)
    table = table.squeeze()[:-1].unsqueeze(0)
    out = net.model.generate(table,
                             max_length=args.maxlen,
                             do_sample=args.do_sample,
                             num_beams=args.num_beams,
                             temperature=args.temperature,
                             top_p=args.top_p,
                             num_return_sequences=args.num_return_sequences)
    candidates = []
    for seq in out:
        decoded = net.tokenizer.decode(seq, skip_special_tokens=True)
        decoded = decoded.split('|||')[1:][0].strip()
        candidates.append(decoded)
    captions = clip_rescoring(args, net, candidates, x[None,:])
    return captions[:args.num_captions]


def captioner(args, net):
    dataset = ImageDataset(folder=args.image_dir)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.nworkers,
                      pin_memory=True,
                      collate_fn=dl_collate_fn,
                      prefetch_factor=2)
    print('Captioning images...')
    for imgs, paths in tqdm(data):
        ctx = [args.context] * len(imgs)
        table, x = net.build_table(imgs.half().to(args.device),
                                   net.perceiver,
                                   ctx=ctx,
                                   indices=net.indices,
                                   indices_data=net.indices_data,
                                   knn=args.knn,
                                   tokenize=clip.tokenize,
                                   device=args.device,
                                   is_image=True,
                                   return_images=True)
        for idx in range(len(table)):
            captions = caption_image([table[idx]], x, args, net)
            result = ''.join(f'{captions[i]}\n' for i in range(len(captions)))
            paths[idx].write_text(result)
        

def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--index_dirs', type=str, default=None)
    parser.add_argument('--context', type=str, default=None)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--knn', type=int, default=5)
    parser.add_argument('--maxlen', type=int, default=64)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--num_return_sequences', type=int, default=250)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--num_captions', type=int, default=5)
    args = parser.parse_args()

    print('Loading indices...')
    indices = []
    indices_data = []
    index_dirs = args.index_dirs.split(',')
    index_dirs = list(filter(lambda t: len(t) > 0, index_dirs))
    for index_dir in index_dirs:
        fname = os.path.join(index_dir, 'args.txt')
        with open(fname, 'r') as f:
            index_args = dotdict(json.load(f))
        
        entries = []
        fname = os.path.join(index_dir, 'entries.txt')
        with open(fname, 'r') as f:
            entries.extend([line.strip() for line in f])

        indices_data.append(entries)
        indices.append(faiss.read_index(glob.glob(f"{index_dir}/*.index")[0]))
    
    print('Loading model...')
    config = dotdict(torch.load(args.config))
    net = retrofit.load_params(config).to(args.device).half()
    net.indices = indices
    net.indices_data = indices_data

    # Generate captions
    captioner(args, net)


if __name__ == '__main__':
    main()