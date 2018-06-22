#!/usr/bin/env python3

import argparse
import pylib.inputdata as inputdata
import pylib.merger as merger
import pylib.util as util
import configobj
import validate
import pylib.log as log
import os.path

if __name__ == "__main__":
    logger = log.setup_logger("main")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        dest="input_path",
        type=str,
        nargs="+",
        help="InputData video file"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Config file."
    )
    parser.add_argument(
        "-i",
        "--iterator",
        default="VideoFrameIterator",
        help="Frame iterator. Default: VideoFrameIterator",
        choices=util.get_all_subclasses_names(inputdata.FrameIterator)
    )
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
        choices=["frame", "mean", "diff", "metric", "merge", "final"],
        help="Which steps to save."
    )
    parser.add_argument(
        "-v",
        "--preview",
        type=str,
        nargs="+",
        default=["final"],
        choices=["frame", "mean", "diff", "metric", "merge", "final"],
        help="Which steps to preview."
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
        default=[],
        help="Set parameters of your merger. Give strings like "
             "<param_name>=<param_value>. Note: If you want to pass a string, "
             "use quotation marks, e.g. param='blah'"
    )

    logger.debug("Parsing command line options.")
    args = parser.parse_args()

    # with open("configspec.config") as configspec_file:
    #     configspec = configspec_file.readlines()

    if args.config and not os.path.exists(args.config):
        logger.error("Config file '{}' does not exist. Falling back to default.".format(
            args.config))
        args.config = "tmp.config"

    logger.debug("Parsing config file.")
    try:
        config = configobj.ConfigObj(args.config, configspec="configspec.config")
    except:
        msg = "Some error occurred during reading of config file. " \
              "Aborting now."
        logger.critical(msg)
        raise ValueError

    valid = config.validate(validate.Validator())
    config.filename = "out.config"
    config.write()

    print(config)
    if not valid:
        msg = "Config file validation failed."
        logger.critical(msg)
        raise ValueError(msg)

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    i = inputdata.InputData(args.input_path, getattr(inputdata, args.iterator))
    m = getattr(merger, args.merger)(i, config)

    # todo: rather implement with config
    m.save = args.save
    m.preview = args.preview

    # todo: implement with config
    if args.name:
        m.name = args.name

    # todo: implement with config
    for param_value_pair in args.parameter:
        assert(param_value_pair.count("=") == 1)
        key, value = param_value_pair.split("=")
        key = key.strip()
        value = value.strip()
        logger.debug("Setting key '{}' to value '{}'='{}'".format(
            key, value, eval(value)))
        setattr(m, key, eval(value))

    m.run()
