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
        default=None,
        choices=["frame", "mean", "diff", "metric", "merge", "final"],
        help="Which steps to save."
    )
    parser.add_argument(
        "-v",
        "--preview",
        type=str,
        nargs="+",
        default=None,
        choices=["frame", "mean", "diff", "metric", "merge", "final"],
        help="Which steps to preview."
    )
    parser.add_argument(
        "-p",
        "--parameter",
        type=str,
        nargs="+",
        default=[],
        help="Set parameters of your merger. Give strings like "
             "<param_name>=<param_value>. To give a list of values, make sure "
             "param_value contains a ',', even when passing only one value, "
             "e.g. 'key=value,'. When passing multiple list members, so "
             "key=value1,value2."
             "We will try to convert each value to"
             " a float (if it contains a dot) or an int. If both fail, we "
             "take it as a string."
    )

    logger.debug("Parsing command line options.")
    args = parser.parse_args()

    # with open("configspec.config") as configspec_file:
    #     configspec = configspec_file.readlines()

    if args.config and not os.path.exists(args.config):
        logger.error("Config file '{}' does not exist. Falling back to "
                     "default.".format(args.config))
        args.config = "tmp.config"

    logger.debug("Loading config file.")
    try:
        config = configobj.ConfigObj(args.config,
                                     configspec="configspec.config")
    except:
        msg = "Some error occurred during reading of config file. " \
              "Aborting now."
        logger.critical(msg)
        raise ValueError

    logger.debug("Validating config file.")
    valid = config.validate(validate.Validator(), preserve_errors=True,
                            copy=True)

    # adapted from https://stackoverflow.com/questions/14345879/
    # answer from user sgt_pats 2017
    for entry in configobj.flatten_errors(config, valid):
        [sectionList, key, error] = entry
        if error == False:
            msg = "The parameter {} was not in the config file\n".format(key)
            msg += "Please check to make sure this parameter is present and " \
                   "there are no mis-spellings."
            logger.critical(msg)
            raise ValueError(msg)

        if key is not None:
            if isinstance(error, validate.VdtValueError):
                optionString = config.configspec[key]
                msg = "The parameter {} was set to {} which is not one of " \
                      "the allowed values\n".format(key, config[key])
                msg += "Please set the value to be in {}".format(optionString)
                logger.critical(msg)
                raise ValueError(msg)

        raise ValueError("Unknown error with validation.")

    logger.debug("Setting cli options to config file.")
    if args.save is not None:
        config["m"]["save"] = args.save
    if args.preview is not None:
        config["m"]["preview"] = args.preview
    if args.name:
        config["m"].name = args.name

    logger.debug("Checking for -p/--parameter options.")

    for param_value_pair in args.parameter:
        if not param_value_pair.count("=") == 1:
            logger.error("Do you want to set a parameter with '{}'? If so, "
                         "this should have exactly one '='! "
                         "Skipping this for now.")
            continue

        keys, value_str = param_value_pair.split("=")
        keys = keys.strip().split(".")
        value_str = value_str.strip()

        if "," in value_str:
            # assuming this is a list
            value_evaluated = []
            for s in value_str.split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    v = float(s) if '.' in s else int(s)
                except ValueError:
                    v = s
                value_evaluated.append(v)
        else:
            try:
                value_evaluated = float(value_str) if '.' in value_str else int(value_str)
            except ValueError:
                value_evaluated = value_str

        this_config = config
        path = ""
        fail = False
        for key in keys[:-1]:
            path += "/" + key
            try:
                this_config = this_config[key]
            except KeyError:
                logger.error("The config item '{}' does not seem to exist. "
                             "Skipping this for now.".format("/".join(keys)))
                fail = True
                break
        this_config[keys[-1]] = value_evaluated

        if fail:
            continue

        logger.debug("Setting config item '{}' to value '{}'='{}' ({})".format(
            '/'.join(keys), value_str, value_evaluated, type(value_evaluated)))

    logger.debug("Validating config file again")
    valid = config.validate(validate.Validator(), preserve_errors=True,
                            copy=True)

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    i = inputdata.InputData(args.input_path, getattr(inputdata, args.iterator))
    m = merger.SimpleMerger(i, config)



    m.run()
