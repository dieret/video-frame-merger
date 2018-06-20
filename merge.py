#!/usr/bin/env python3

import os
import argparse
import pylib.input as input
import pylib.merger as merger
import pylib.util as util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        dest="input_path",
        type=str,
        nargs="+",
        help="Input video file")
    parser.add_argument(
        "-i",
        "--iterator",
        default="VideoFrameIterator",
        help="Frame iterator. Default: VideoFrameIterator",
        choices=util.get_all_subclasses_names(input.FrameIterator))

    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Name (will e.g. become output folder name)"
    )

    parser.add_argument(
        "-s",
        "--save",
        type=str,
        nargs="+",
        default=["final"],
        choices=["mean", "diff", "metric", "merge", "final"],
        help="Which steps to save."
    )

    parser.add_argument(
        "-m",
        "--merger",
        default="SimpleMerger",
        help="Merger. This is what actually merges all frames. "
             "Default: SimpleMerger.",
        choices=util.get_all_subclasses_names(merger.Merger)
    )

    # todo: implement merger.save

    args = parser.parse_args()

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    inpt = input.Input(args.input_path, getattr(input, args.iterator))
    m = getattr(merger, args.merger)(inpt, name=args.name)
    m.save = args.save

    m.run()
