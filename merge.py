#!/usr/bin/env python3

import os
import argparse
import pylib.input as input
import pylib.merger as merger


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
        help="Frame iterator. Currently 3 options: "
             "VideoFrameIterator, SinglFramesIterator, "
             "BurstFrameIterator.")
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Show picture upon completion."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join("out", "out.png"),
        help="Output path."
    )

    args = parser.parse_args()

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    inpt = input.Input(args.input_path, getattr(input, args.iterator))
    m = merger.Merger1(inpt)
    m.calc_merged()

    if args.show:
        m.show_image(m.merged_image)

    m.save_image(m.merged_image, path=args.output)
