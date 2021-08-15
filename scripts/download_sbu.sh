#!/bin/bash

download_dir='/data'

let npts=1000000
let dev_size=5000
let train_size=npts-dev_size

wget -P ${download_dir} https://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz
tar -zxvf ${download_dir}/SBUCaptionedPhotoDataset.tar.gz -C ${download_dir}

mkdir "${download_dir}/sbu_train"
mkdir "${download_dir}/sbu_dev"

datapath="${download_dir}/dataset"
cat "${datapath}/SBU_captioned_photo_dataset_captions.txt" | head -${train_size} > "${download_dir}/train_captions.txt"
cat "${datapath}/SBU_captioned_photo_dataset_captions.txt" | tail -${dev_size} > "${download_dir}/dev_captions.txt"

split -l 10 --additional-suffix=.txt "${download_dir}/train_captions.txt" "${download_dir}/sbu_train/"
split -l 1 --additional-suffix=.txt "${download_dir}/dev_captions.txt" "${download_dir}/sbu_dev/"