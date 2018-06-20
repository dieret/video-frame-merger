#!/usr/bin/env python3

def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])

def get_all_subclasses_names(cls):
    return [c.__name__ for c in get_all_subclasses(cls)]
