#!/usr/bin/env bash

# This script will run all examples from the readme

commands=$(grep "    ./merge.py -n examples/" readme.md | sed 's/    //g')

while read -r command; do
    echo "***************************************************************"
    echo $command
    echo "***************************************************************"
    $command
done <<< "${commands}"