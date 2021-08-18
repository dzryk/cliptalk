import sys
import os
import os.path
import random
import argparse

from pathlib import Path

import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from conditional text datasets.""")
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=100000)
parser.add_argument(
    "--compression", 
    dest="compression", 
    action="store_true",
    help="Creates compressed .tar.gz files instead of uncompressed .tar files."
    )
parser.add_argument(
    "--text_keys", 
    type=str, 
    default="ctx,cap",
    help="Comma separated WebDataset dictionary keys for text."
    )
parser.add_argument(
    "--shards", 
    default="./shards", 
    help="directory where shards are written"
)
parser.add_argument(
    "--shard_prefix", 
    default="ds_", 
    help="prefix of shards' filenames created in the shards-folder"
)
parser.add_argument(
    "--data",
    default="./data",
    help="directory path containing text files.",
)
args = parser.parse_args()

assert len(args.text_keys.split(',')) == 2, 'Too many arguments provided'
assert args.maxsize > 10000000
assert args.maxcount < 1000000

context_key, caption_key = tuple(args.text_keys.split(','))

if not os.path.isdir(os.path.join(args.data)):
    print(f"{args.data}: should be directory containing text files", file=sys.stderr)
    print(f"or subfolders containing text files", file=sys.stderr)
    sys.exit(1)

os.makedirs(Path(args.shards), exist_ok=True)

def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

path = Path(args.data)
text_files = [*path.glob('**/*.txt')]
text_files = {text_file.stem: text_file for text_file in text_files}
text_total = len(text_files)

context_files = [*path.glob('**/*.ctx')]
context_files = {context_file.stem: context_file for context_file in context_files}
context_total = len(context_files)

print('Found {:,} textfiles and {:,} context files.'.format(text_total, context_total))

keys = (context_files.keys() & text_files.keys())

text_files = {k: v for k, v in text_files.items() if k in keys}
context_files = {k: v for k, v in context_files.items() if k in keys}

total_pairs = len(keys)
keys = list(keys)

indexes = list(range(total_pairs))
random.shuffle(indexes)

# This is the output pattern under which we write shards.
pattern = os.path.join(args.shards, args.shard_prefix + f"%06d.tar" + (".gz" if args.compression else ''))

with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
    for i in indexes:
        with open(context_files[keys[i]], "rb") as txtstream:
            context = txtstream.read()
        with open(text_files[keys[i]], "rb") as txtstream:
            text = txtstream.read()

        ds_key = "%09d" % i

        sample = {
            "__key__": ds_key,
            context_key: context,
            caption_key: text
        }
        sink.write(sample)