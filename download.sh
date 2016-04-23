#!/usr/bin/env bash

sudo chown -R ubuntu /mnt 
sudo chmod 777 /mnt

DIR="$(dirname $0)"/data

rm -rf /mnt/heartvol_readonly/
mkdir -p /mnt/heartvol_readonly/
ln -sf /mnt/heartvol_readonly/ $DIR

for file in train.csv; do
    s3cmd get --force "s3://heartvol/data/$file" "/mnt/heartvol_readonly/$file"
done

for file in train.zip validate.zip challenge_training.zip challenge_validation.zip challenge_online.zip Sunnybrook_Cardiac_MR_Database_ContoursPart1.zip Sunnybrook_Cardiac_MR_Database_ContoursPart2.zip Sunnybrook_Cardiac_MR_Database_ContoursPart3.zip; do
    s3cmd get --force "s3://heartvol/data/$file" "/mnt/heartvol_readonly/$file"
    unzip -d /mnt/heartvol_readonly "/mnt/heartvol_readonly/$file"
    rm "/mnt/heartvol_readonly/$file"
done
