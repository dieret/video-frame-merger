#!/bin/bash

set -e

for file in out/merged_*.png; do
    echo "${file}"
    name=$(basename ${file})
    convert -quality 90 -resize 20% "${file}" "out/lq_${name}"
done
