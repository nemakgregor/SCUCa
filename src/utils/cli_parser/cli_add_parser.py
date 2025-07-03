import argparse
from typing import Dict

from utils.cli_parser.cli_options import CLI_OPT_FNAMES, CLIOption


def _add_parser_option(
    parser: argparse.ArgumentParser,
    opt: CLIOption,
) -> None:
    args = []
    for arg in [opt.long_name, opt.short_name]:
        if arg is not None:
            args.append(arg)
    kargs = {
        fn: getattr(opt, fn)
        for fn in CLI_OPT_FNAMES
        if fn not in ["long_name", "short_name"]
    }
    kwargs = {k: v for k, v in kargs.items() if v is not None}
    parser.add_argument(*args, **kwargs)


def setup_parser(parser: argparse.ArgumentParser, options: Dict) -> None:
    for opt in options["common"]:
        _add_parser_option(parser, opt)
