#!/usr/bin/env python3

import os
import sys


# todo:docstring
def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])


# todo: docstring
def get_all_subclasses_names(cls):
    return [c.__name__ for c in get_all_subclasses(cls)]


# todo: docstring
def new_size(old_size, max_size):
    # note: don't use unpacking, because the tuples
    # might contain more entries (e.g. for number of colors etc.)
    old_height = old_size[0]
    old_width = old_size[1]
    max_height = max_size[0]
    max_width = max_size[1]
    if max_width and max_height:
        new_size = (max_height, max_width)
    elif max_width:
        if max_width >= old_width:
            new_size = (old_height, old_width)
        else:
            scale = max_width/old_width
            new_size = (scale * old_height, scale * old_width)
    elif max_height:
        if max_height >= old_height:
            new_size = (old_height, old_width)
        else:
            scale = max_height/old_height
            new_size = (scale * old_height, scale * old_width)
    else:
        new_size = (old_height, old_width)
    return new_size


# todo: docstring
def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
