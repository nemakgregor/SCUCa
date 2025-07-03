"""
Executable script for SCUC project
"""

import argparse
import logging
import time
import pathlib
# import sys

from utils.cli_parser.cli_options import CLI_OPTIONS
from utils.cli_parser.cli_add_parser import setup_parser
from utils.defaults import DEFAULTS
from io.input.read_input import read_scenario
# from instances.instance_structure import UnitCommitmentScenario

logger = logging.getLogger("ase_production_planning")
logging.basicConfig()

# Define the local path
LOCAL_PATH = pathlib.Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="SCUC project",
    )
    setup_parser(parser, CLI_OPTIONS)

    args = parser.parse_args()

    # Set log level
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    logger.debug(" Parsed arguments: %s", args)

    if not args.input_data:
        logger.warning(" No input data provided. Test case14 will be used.")
        args.input_data = DEFAULTS.DEFAULT_CASE
    else:
        if args.download_case:
            logger.info(" Downloading test case: %s", args.download_case)
        else:
            logger.info(" Using provided case: %s", args.input_data)

    logger.info(" Case name: %s", args.input_data)

    # Check if input data path exists
    input_path = (
        LOCAL_PATH / pathlib.Path(DEFAULTS.INPUT_PATH) / pathlib.Path(args.input_data)
    )

    if not input_path.exists():
        logger.warning("Input data directory does not exist: %s", input_path)
    else:
        logger.info("Input data directory exists: %s", input_path)

    try:
        # Read the input data
        logger.info(" Reading input data from: %s", input_path)
        scenario = read_scenario(str(input_path))

        # Log the scenario details
        logger.info(" Scenario loaded successfully.")
        logger.debug(" Scenario details: %s", scenario)

    except Exception as e:
        logger.error("Error reading input data: %s", e)
        return


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"SCUC run time: {end_time - start_time:.2f} seconds\n")
