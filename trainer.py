import importlib
import json
import glob
import faiss
import os
import torch
import numpy as np
import pytorch_lightning as pl

from CLIP import clip
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--train_datadir', type=str)
    parser.add_argument('--dev_datadir', type=str)
    parser.add_argument('--index_dirs', type=str, default=None)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--gpt', type=str, default='gpt2-large')
    parser.add_argument('--ft', type=str, default='bias')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--knn', type=int, default=5)
    parser.add_argument('--maxlen', type=int, default=64)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--val_after_n_epochs', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=1e5)
    parser.add_argument('--save_top_k', type=int, default=10)
    parser.add_argument('--lrate', type=float, default=3e-4)
    args = parser.parse_args()

    # Set wandb cache dir
    os.environ['WANDB_CACHE_DIR'] = args.logdir

    # Import the task
    task = importlib.import_module(f'tasks.{args.task}')

    # Load indices
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
    preprocess = clip.load(args.clip_model, jit=False)[1]

    # Train model
    wandb_logger = WandbLogger(
        save_dir=args.logdir,
        project='cliptalk',
        log_model='all')
    ckpt_callback = ModelCheckpoint(
        monitor='vloss',
        mode='min',
        filename='-{epoch:02d}-{vloss:.3f}',
        save_top_k=args.save_top_k)
    datamodule = task.DataModule(
        train_datadir=args.train_datadir,
        dev_datadir=args.dev_datadir,
        batch_size=args.batch_size,
        nworkers=args.nworkers)
    net = model.Model(args, indices=indices, indices_data=indices_data)
    trainer = pl.Trainer(
        default_root_dir=args.logdir,
        logger=wandb_logger,
        gpus=args.gpus,
        precision=args.precision,
        tpu_cores=args.tpu_cores,
        max_steps=args.tmax,
        callbacks=[ckpt_callback],
        check_val_every_n_epoch=args.val_after_n_epochs)
    trainer.fit(net, datamodule)


if __name__ == '__main__':
    main()