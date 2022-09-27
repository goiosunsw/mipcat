#!/bin/bash
echo "Start"
for f in $(find $1 -iname "*.textgrid"); do
    echo " $f";
    if  file "$f" | grep --quiet -i 'utf-16' 
    then
        fbak="${f}.utf16.bak"
        echo "  UTF16";
        cp $f $fbak
        iconv -f utf-16 -t utf-8 $fbak | sed s/$( echo -ne '\u266f')/#/ > $f
    fi
done