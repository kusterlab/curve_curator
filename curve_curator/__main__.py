# ############# #
# Curve Curator #
# ############# #
#
# A Complete analysis pipeline of dose-response curves including fitting, filtering, and visualization.
#
# usage:
# python -m curve_curator [-h] [-r [RANDOM]] [-b] <PARAM PATH>
#
# Florian P. Bayer - 2024
#

from pathlib import Path
import argparse

from . import user_interface as ui
from . import data_parser
from . import data_simulator
from . import quantification
from . import thresholding
from . import dashboard
from . import quality_control
from .__init__ import __version__


def main():
    # Build a command line parser for parsing multiple config files
    command_line = argparse.ArgumentParser(
        description='CurveCurator',
        epilog="Complete analysis pipeline of dose-response curves including fitting, filtering, and visualization.\n\nFPB-2023")
    command_line.add_argument(
        "-b", "--batch",
        default=False,
        action='store_true',
        help="Run a batch process with a file containing all the parameter file paths.")
    command_line.add_argument(
        "-f", "--fdr",
        default=False,
        action='store_true',
        help="Estimate FDR based on target decoy approach. Estimating the FDR will double the run time.")
    command_line.add_argument(
        "-m", "--mad",
        default=False,
        action='store_true',
        help="Perform the medium absolute deviation (MAD) analysis to detect outliers")
    command_line.add_argument(
        "-r", "--random",
        type=int,
        nargs="?",
        help="Run the pipeline with <N> random values for H0 simulation.")
    command_line.add_argument(
        dest="path",
        metavar="<PATH>",
        type=str,
        help="Relative path to the config.toml or batch.txt file to run the pipeline.")

    # Parse the terminal arguments
    args = command_line.parse_args()

    # Welcome
    ui.breakline()
    ui.welcome()
    ui.breakline()

    # Handle batch process
    if args.batch:
        with open(args.path) as f:
            toml_files = f.read().splitlines()
        ui.message(f' * Batch process detected with {len(toml_files)} toml files.')
        ui.breakline()
    else:
        toml_files = [args.path]

    # Go through all toml files and run the scripts
    for i, tf in enumerate(toml_files):
        # Set up the logger
        ui.setup_logger(Path(tf).parent, name=i)
        ui.message(f' * Executing CurveCurator pipeline version {__version__}.')

        # Make a counter in batch mode only for the terminal
        if args.batch:
            ui.message(f' * Processing {i+1} of {len(toml_files)} data sets.', terminal_only=True)

        # Check the input file is a toml file
        if not ui.is_toml_file(tf):
            ui.error(f' * The given file is not a TOML parameter file !\n * If it\'s a batch file make sure you activate the batch mode with --batch.')
            ui.doneline()
            continue

        # Load config
        config = ui.load_toml(tf, random_mode=bool(args.random))
        config = ui.set_default_values(config)

        # In the random mode sample random data (H0=True)
        if args.random is not None:
            data_simulator.sample(config, n=args.random)

        # Run the pipeline for target curves
        data = data_parser.load(config)
        data = quantification.run_pipeline(data, config=config)
        data = thresholding.apply_significance_thresholds(data, config=config)

        # Check Quality
        if args.mad:
            quality_control.mad_analysis(data, config=config)

        # Run the pipeline for decoy curves in FDR mode
        if args.fdr:
            decoys = data_simulator.get_decoys(data, config=config)
            decoys = quantification.run_pipeline(decoys, config=config, decoy_mode=True)
            decoys = thresholding.apply_significance_thresholds(decoys, config=config)
            fdr = thresholding.estimate_fdr(data, decoys, config=config)
            decoys.to_csv(config['Paths']['decoys_file'], sep='\t', index=False)

        # Save curve file
        data.to_csv(config['Paths']['curves_file'], sep='\t', index=False)

        # Plot the data in the dashboard
        dashboard.render(data, config=config)

        # Done
        ui.doneline()


if __name__ == '__main__':
    main()
