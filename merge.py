#!/usr/bin/env python3

import argparse
import pylib.inputdata as inputdata
import pylib.merger as merger
import pylib.util as util
import configobj
import validate
import pylib.log as log
import os.path
import logging
from typing import List, Any


def get_cli_options(logger: logging.Logger):
    """ Return command line options. """
    logger.debug("Parsing command line options.")
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
        help="Path to config file."
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
        help="Which steps to save."
    )
    parser.add_argument(
        "-v",
        "--preview",
        type=str,
        nargs="+",
        default=None,
        help="Which steps to preview."
    )
    parser.add_argument(
        "-p",
        "--parameter",
        type=str,
        nargs="+",
        default=[],
        help="Set parameters of your merger. Give strings like "
             "<param_name>=<param_value>. "
             "To specify subsections, use '.', e.g. 'section1.section2.key'. "
             "To give a list of values, make sure "
             "param_value contains a ',', even when passing only one value, "
             "e.g. 'key=value,'. When passing multiple list members, so "
             "key=value1,value2."
             "We will try to convert each value to"
             " a float (if it contains a dot) or an int. If both fail, we "
             "take it as a string."
    )

    return parser.parse_args()


def load_config_file(path: str, logger: logging.Logger) -> configobj.ConfigObj:
    if path and not os.path.exists(args.config):
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

    config = validate_config_file(config, logger)

    return config


def validate_config_file(config: configobj.ConfigObj,
                         logger: logging.Logger) -> configobj.ConfigObj:
    """ Validate config file, i.e. check that everything is set properly. """

    logger.debug("Validating config file.")

    valid = config.validate(validate.Validator(), preserve_errors=True,
                            copy=True)

    # adapted from https://stackoverflow.com/questions/14345879/
    # answer from user sgt_pats 2017
    for entry in configobj.flatten_errors(config, valid):
        [_, key, error] = entry
        if not error:
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

    return config


def interpret_setter(string: str, logger: logging.Logger) -> (List[str], Any):
    """ Interpret strings such as 'sec1.mykey=5' bt splitting them up in a key 
    and a evaluated value (float, int, string or list thereof). """
    if not string.count("=") == 1:
        logger.error("Do you want to set a parameter with '{}'? If so, "
                     "this should have exactly one '='!")
        raise ValueError

    keys, value_str = string.split("=")
    keys = keys.strip()
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

    return keys, value_evaluated


def set_config_option(config: configobj.ConfigObj, path: List[str], value: Any,
                      logger: logging.Logger):
    """ Example: set_config_option(config, ['sec1', 'sec2', 'key'], 5, logger) 
    will do config['sec1']['sec2']['key'] = 5 and return the new config 
    object. """
    this_config = config
    _path = ""
    for key in path[:-1]:
        _path += "/" + key
        try:
            this_config = this_config[key]
        except KeyError:
            logger.error("The config item '{}' does not "
                         "seem to exist.".format(_path))
            raise ValueError
    this_config[path[-1]] = value
    logger.debug("Setting config item '{}' to value '{}' ({})".format(
            '/'.join(path), value_evaluated, type(value_evaluated)))
    return config


if __name__ == "__main__":
    logger = log.setup_logger("main")

    args = get_cli_options(logger)
    config = load_config_file(args.config, logger)

    logger.debug("Setting cli options to config file.")
    if args.save is not None:
        config["m"]["save"] = args.save
    if args.preview is not None:
        config["m"]["preview"] = args.preview
    if args.name:
        config["m"].name = args.name

    logger.debug("Checking for -p/--parameter options.")

    for string in args.parameter:
        try:
            keys, value_evaluated = interpret_setter(string, logger)
        except ValueError:
            logger.error("Skipping this issue for now.")
            continue
        path = keys.split(".")
        try:
            config = set_config_option(config, path, value_evaluated, logger)
        except ValueError:
            logger.error("Skipping this issue for now.")
            continue

    config = validate_config_file(config, logger)

    logger.info("Saving config file to tmp.config.")
    config.filename = "tmp.config"
    config.write()

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    i = inputdata.InputData(args.input_path, getattr(inputdata, args.iterator))
    m = merger.SimpleMerger(i, config)

    m.run()

    logger.info("Config file was saved to {}.".format(config.filename))
