#!/bin/bash

mkdir -p burst
for file in giflibrary/*.gif; do
    # echo $file
	# note: coalesce: get full frames, not incremental differenes 
	convert "${file}" -coalesce "${file%.gif}.png"
	mv ${file%.gif}*.png burst/
done

