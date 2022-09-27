#!/bin/bash

ext_ch_no=6

findpath=$1

for f in $(find $findpath -iname "*.wav"); do 
    info=$(soxi $f)
    nchan=$(echo "$info" | sed -E -ne '/Channels/s/.*([[:digit:]]+)/\1/gp'); 
    if ((nchan > 2)); then
        dest=${f/$1\//}
        destdir=$(dirname $dest)
        mkdir -p $destdir
        sox $f $dest remix $ext_ch_no
    else
        echo $f Normal audio
    fi

done
