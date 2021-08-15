import json

from collections import defaultdict
from argparse import ArgumentParser


def save_text(datadir, split):
    """Loads the captions json, extracts the captions and stores them."""
    path = f'{datadir}/personality_captions/{split}.json'
    with open(path, 'r') as f:
        captions = json.load(f)
    keys = range(len(captions))
    for k in keys:
        ctxfname = f'{str(k).zfill(12)}.ctx'
        txtfname = f'{str(k).zfill(12)}.txt'
        ctx = captions[k]['personality']
        txt = captions[k]['comment']
        ctxpath = f'{datadir}/personality_captions/{split}/{ctxfname}'
        txtpath = f'{datadir}/personality_captions/{split}/{txtfname}'
        with open(ctxpath, 'w') as f:
            f.write(ctx + '\n')
        with open(txtpath, 'w') as f:
            f.write(txt + '\n')


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    args = parser.parse_args()

    save_text(args.datadir, 'train')
    save_text(args.datadir, 'val')


if __name__ == '__main__':
    main()