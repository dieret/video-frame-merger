#!/bin/bash

set -e

for file in $@; do
    echo "${file}"
    name=$(basename ${file})
    dirpath=$(dirname ${file})
    convert -quality 90 -resize 20% "${file}" "${dirpath}/lq_${name}"
done
