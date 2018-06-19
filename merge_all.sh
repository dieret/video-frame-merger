#!/bin/bash

for gif in data/giflibrary/*.gif; do
    name=$(basename "${gif}")
    ./merge.py "${gif}" -o out/${name%.gif}.png
done