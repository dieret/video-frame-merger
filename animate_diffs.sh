#!/bin/bash

set -e

convert $@ $(dirname $1)/animated.gif
