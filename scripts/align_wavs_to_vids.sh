#!/bin/bash

rootdir=$1

for dir in ls -d $rootdir/*/; do 
    find  $dir -iname "*.wav" -print0 | while read -d $'\0' wf 
    do
        find  $dir -iname "*.mp4" -print0 | while read -d $'\0' vf 
        do
            vfn=${vf//\//_}
            wfn=${wf//\//_}
            ret=$(python -m clemotion.align.align_keypoints -c 5 $vf $wf > ${vfn}__${wfn}_align.txt )
        done
    done
done
