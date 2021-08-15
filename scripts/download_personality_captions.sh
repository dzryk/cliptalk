#!/bin/bash

download_dir='/data'

parlai display_data -t personality_captions --datapath=${download_dir}
mkdir ${download_dir}/personality_captions/train
mkdir ${download_dir}/personality_captions/val

python3 personality_captions.py --datadir=${download_dir}