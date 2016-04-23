#!/usr/bin/env bash

rm -rf /mnt/lv/
mkdir -p /mnt/lv/
mkdir /mnt/lv/data/
mkdir /mnt/lv/labels

s3cmd get --force s3://heartvol/sunnybrook/train_img.zip /mnt/lv/data/train_img.zip
unzip -d /mnt/lv/data/ /mnt/lv/data/train_img.zip
rm -rf /mnt/lv/data/train_img.zip

s3cmd get --force s3://heartvol/sunnybrook/train_con.zip /mnt/lv/labels/train_con.zip
unzip -d /mnt/lv/labels/ /mnt/lv/labels/train_con.zip
rm -rf /mnt/lv/labels/train_con.zip
