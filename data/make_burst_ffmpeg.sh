#!/bin/bash

mkdir -p burst
for file in giflibrary/*.gif; do
    # echo $file
	# note: coalesce: get full frames, not incremental differenes 
	ffmpeg -i "${file}" ${file%.gif}-%03d.png
	mv ${file%.gif}*.png burst/
done

