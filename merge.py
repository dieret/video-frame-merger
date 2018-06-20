#!/usr/bin/env python3

import argparse
import pylib.inputdata as inputdata
import pylib.merger as merger
import pylib.util as util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        dest="input_path",
        type=str,
        nargs="+",
        help="InputData video file")
    parser.add_argument(
        "-i",
        "--iterator",
        default="VideoFrameIterator",
        help="Frame iterator. Default: VideoFrameIterator",
        choices=util.get_all_subclasses_names(inputdata.FrameIterator))

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
    parser.add_argument(
        "-p",
        "--parameter",
        type=str,
        nargs="+",
        help="Set parameters of your merger. Give strings like "
             "<param_name>=<param_value>."
    )

    args = parser.parse_args()

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    i = inputdata.InputData(args.input_path, getattr(inputdata, args.iterator))
    m = getattr(merger, args.merger)(i)
    m.save = args.save

    if args.name:
        m.name = args.name

    for param_value_pair in args.parameter:
        assert(param_value_pair.count("=") == 1)
        key, value = param_value_pair.split("=")
        key = key.strip()
        value = value.strip()
        m._logger.debug("Setting key '{}' to value '{}'='{}'".format(
            key, value, eval(value)))
        setattr(m, key, eval(value))

    m.run()
