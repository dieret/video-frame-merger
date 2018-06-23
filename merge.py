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
             "take it as a string. For lists, you can also write '+=' or '-=' "
             "to add or remove values from the list. "
    )

    return parser.parse_args()


def load_config_file(path: str, logger: logging.Logger) -> configobj.ConfigObj:
    if path and not os.path.exists(args.config):
        logger.error("Config file '{}' does not exist. Falling back to "
                     "default.".format(args.config))
        args.config = "tmp.config"

    logger.debug("Loading config file.")

    try:
        path = os.path.join(util.get_script_path(), "configspec.config")
        config = configobj.ConfigObj(args.config, configspec=path)
    except:
        msg = "Some error occurred during reading of config file. " \
              "Aborting now."
        logger.critical(msg)
        raise ValueError

    config = validate_config_file(config, logger)

    return config


def validate_config_file(config: configobj.ConfigObj,
                         logger: logging.Logger) -> configobj.ConfigObj:
    """ Validate config file, i.e. check that everything is set properly.
    This also sets all default values."""

    logger.debug("Validating config file.")

    # 'copy' parameter of config.validate: Also copy all comments and default
    # values from the configspecs to the config file (to be written out later)?
    # Note: The config object might be completely empty, because only the first
    # Run of this methods sets them from the defaults if the user did not
    # specify a config file.
    # However, this method gets called 2 times (one time before and one time
    # after the handling of the --parameter options),
    # Unfortunately, if you do copy one time, an additional copy=False won't
    # bring it back. So we start with copy=False and if r.config.copy_all
    # is set to true, copy it the second time this function is called.
    try:
        copy_all = config["r"]["config"]["copy_all"]
    except KeyError:
        copy_all = False

    valid = config.validate(validate.Validator(), preserve_errors=True,
                            copy=copy_all)

    # adapted from https://stackoverflow.com/questions/14345879/
    # answer from user sgt_pats 2017
    # todo: This might need some better handling.
    for entry in configobj.flatten_errors(config, valid):
        [path, key, error] = entry
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
            elif error:
                msg = "Validation error (section='{}', key={}): {}".format('/'.join(path), key, error)
                logger.error(msg)
                raise ValueError(msg)

    return config


# todo: needed: crack down on name shadowing


def interpret_setter(string: str, logger: logging.Logger) -> (List[str], str, Any):
    """ Interpret strings such as 'sec1.mykey=5' bt splitting them up in a key 
    and a evaluated value (float, int, string or list thereof). """
    if not string.count("=") == 1:
        logger.error("Do you want to set a parameter with '{}'? If so, "
                     "this should have exactly one '='!")
        raise ValueError

    keys, value_str = string.split("=")
    keys = keys.strip()

    if keys.endswith("+"):
        keys = keys[:-1]
        setter = "+"
    elif keys.endswith("-"):
        keys = keys[:-1]
        setter = "-"
    else:
        setter = "="

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

    path = keys.split(".")

    logger.debug("Path='{}', setter='{}', value='{}'.".format(
        path, setter, value_evaluated))

    return path, setter, value_evaluated


def set_config_option(config: configobj.ConfigObj, path: List[str],
                      setter: str, value: Any, logger: logging.Logger):
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
            msg = "The config item '{}' does not seem to exist.".format(_path)
            logger.error(msg)
            raise ValueError
    previous_value = this_config[path[-1]]
    if setter == "=":
        this_config[path[-1]] = value
    elif setter in ["+", "-"]:
        if not isinstance(value, list):
            msg = "Since you are using '+=' or '-=' with the -p/--parameter" \
                  " option, it looks like you want to add/remove an item " \
                  "from default value. However the default value is not a " \
                  "list! Maybe you forgot a trailing ',' (even necessary " \
                  "when supplying 0 or 1 list items)?"
            logger.error(msg)
            raise ValueError(msg)
        if setter == "+":
            for v in value:
                if v in previous_value:
                    msg = "Since you are using '+=' with the -p/--parameter" \
                          " option, it looks like you want to add an item " \
                          "from default value. However the default value " \
                          "already contains your value '{}'!".format(v)
                    logger.warning(msg)
                else:
                    this_config[path[-1]].append(v)
        if setter == "-":
            for v in value:
                if not v in previous_value:
                    msg = "Since you are using '-=' with the -p/--parameter" \
                          " option, it looks like you want to remove an item " \
                          "from default value. However the default value " \
                          "does not even contain your value '{}'!".format(v)
                    logger.warning(msg)
                else:
                    this_config[path[-1]].remove(v)
    else:
        msg = "Unknown setter. This is a programming error. Please contact " \
              "the developers."
        logger.critical(msg)
        raise ValueError(msg)

    logger.debug("Set config item '{}' to value '{}' ({})".format(
            '/'.join(path), this_config[path[-1]], type(this_config[path[-1]])))
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

    logger.debug("Checking for -p/--parameter options.")

    for string in args.parameter:
        try:
            path, setter, value_evaluated = interpret_setter(string, logger)
        except ValueError:
            logger.error("Skipping this issue for now.")
            continue
        try:
            config = set_config_option(config, path, setter, value_evaluated, logger)
        except ValueError:
            logger.error("Skipping this issue for now.")
            continue

    config = validate_config_file(config, logger)

    if config["r"]["config"]["copy"]:
        logger.info("Saving config file to tmp.config.")
        config.filename = "tmp.config"
        config.write()
    else:
        logger.info("Saving of config file DISABLED.")

    logger.debug("Here's the full config file: ")
    logger.debug(config)

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    logger.debug("Init input data.")
    i = inputdata.InputData(args.input_path, getattr(inputdata, args.iterator))

    logger.debug("Init merger.")
    m = merger.SimpleMerger(i, config)
    if args.name:
        m.name = args.name

    logger.debug("Run! Here we go!")
    m.run()
