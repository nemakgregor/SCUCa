from collections import namedtuple

CLI_OPT_FNAMES = [
    "long_name",
    "short_name",
    "type",
    "default",
    "choices",
    "action",
    "required",
    "help",
]

CLIOption = namedtuple("CLIOption", CLI_OPT_FNAMES)

CLI_OPTIONS = {
    "common": [
        CLIOption(
            long_name="--input-data",
            short_name="-id",
            type=str,
            default=None,
            choices=None,
            action=None,
            required=False,
            help="dir with input data",
        ),
        CLIOption(
            long_name="--verbose",
            short_name="-v",
            type=None,
            default=0,
            choices=None,
            action="count",
            required=False,
            help="increase verbosity level",
        ),
        CLIOption(
            long_name="--download-case",
            short_name="-dc",
            type=None,
            default=False,
            choices=None,
            action="store_true",
            required=False,
            help="flag to download case from UnitCommitment.jl",
            # https://anl-ceeesa.github.io/UnitCommitment.jl/0.4/guides/instances/#OR-LIB/UC
            # https://axavier.org/UnitCommitment.jl/0.4/instances/
        ),
    ]
}
