#!/bin/bash

download_dir='/data'

let npts=12423374
let dev_size=5000
let train_size=npts-dev_size

wget -P ${download_dir} https://storage.googleapis.com/conceptual_12m/cc12m.tsv

mkdir "${download_dir}/cc12m_train"
mkdir "${download_dir}/cc12m_dev"

cat "${download_dir}/cc12m.tsv" | head -${train_size} | cut -f2 -d$'\t' > "${download_dir}/train_captions.txt"
cat "${download_dir}/cc12m.tsv" | tail -${dev_size} | cut -f2 -d$'\t' > "${download_dir}/dev_captions.txt"

split -l 100 --additional-suffix=.txt "${download_dir}/train_captions.txt" "${download_dir}/cc12m_train/"
split -l 1 --additional-suffix=.txt "${download_dir}/dev_captions.txt" "${download_dir}/cc12m_dev/"