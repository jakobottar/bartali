import configargparse
import argparse
import pathlib
import datetime
import yaml
import yaml.representer
import os
import functools
import itertools
from filelock import FileLock

__all__ = [
    "ExParser",
    "simpleroot",
]


TIME_FORMAT_DIR = "%Y-%m-%d-%H-%M-%S"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DIR_FORMAT = "{num}"
EXT = "yaml"
PARAMS_FILE = "params." + EXT
FOLDER_DEFAULT = "exman"
RESERVED_DIRECTORIES = {"runs", "index", "tmp", "marked"}


def yaml_file(name):
    return name + "." + EXT


def simpleroot(__file__):
    return pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / FOLDER_DEFAULT


def represent_as_str(self, data, tostr=str):
    return yaml.representer.Representer.represent_str(self, tostr(data))


def register_str_converter(*types, tostr=str):
    for T in types:
        yaml.add_representer(T, functools.partial(represent_as_str, tostr=tostr))


register_str_converter(pathlib.PosixPath, pathlib.WindowsPath)


def str2bool(s):
    true = ("true", "t", "yes", "y", "on", "1")
    false = ("false", "f", "no", "n", "off", "0")

    if s.lower() in true:
        return True
    elif s.lower() in false:
        return False
    else:
        raise argparse.ArgumentTypeError(
            s, f"bool argument should be one of {str(true + false)}"
        )


class ExParser(configargparse.ArgumentParser):
    def __init__(
        self,
        *args,
        args_for_setting_config_path=("--config",),
        **kwargs,
    ):

        super().__init__(
            *args,
            args_for_setting_config_path=args_for_setting_config_path,
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            ignore_unknown_config_file_keys=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            **kwargs,
        )

    def parse_known_args(self, *args, **kwargs):
        return super().parse_known_args(*args, **kwargs)
