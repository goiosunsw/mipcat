#!/bin/bash

findpath=$1

for f in $(find $findpath -iname "*.mov" -o -iname "*.mp4"); do 
    echo $f
    sr=$(ffmpeg -i "$f" 2>&1 | sed -E -ne '/Audio/s/.*\ ([0-9]+)\ Hz.*/\1/p')
    dest=${f/..\/}
    dest=${dest%.*}.wav
    destdir=$(dirname $dest)
    mkdir -p $destdir
    ffmpeg -i $f -vn -acodec pcm_s16le -ar $sr -ac 1 $dest
done
