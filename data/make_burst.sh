#!/bin/bash

mkdir -p burst
for file in *.gif; do
	convert "${file}" "${file%.gif}.png"
	mv ${file%.gif}*.png burst/
done
