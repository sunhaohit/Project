#!/bin/bash

for file in ./dataset/*; do
    echo "Found a dir: $file"
    cp $file/point_sample/ply-10000.ply mydataset/
    folder_name=${file##*/}
    mv mydataset/ply-10000.ply mydataset/$folder_name.ply
done