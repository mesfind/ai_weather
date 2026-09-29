"""
MOMP Main Entry Script
----------------------
This script serves as the primary execution point for the Metrics for Onset of 
Monsoon Precipitation (MOMP) package. It orchestrates the calculation of skill 
scores and the generation of spatial performance maps.

Usage:
    momp-run -p demo/et/config_et_det.in --mode prob
    momp-run --mode det -v
    python -m momp.driver -p my_config.in
"""

import argparse
import logging
import sys
import traceback

from momp.lib.loader import get_cfg, get_setting
from momp.app.bin_skill_score import skill_score_in_bins
from momp.app.spatial_far_mr_mae import spatial_far_mr_mae_map
from momp.utils.printing import print_momp_banner

logger = logging.getLogger(__name__)


def run_momp(cfg=None, setting=None):
    """
    Executes the standard MOMP evaluation workflow.

    Args:
        cfg: The loaded configuration object.
        setting: The environment and directory settings.
    """
    cfg = cfg if cfg is not None else get_cfg()
    setting = setting if setting is not None else get_setting()

    print_momp_banner(cfg)

    logger.info("Starting MOMP Workflow...")
    logger.info(f"Mode: {'PROBABILISTIC' if cfg.probabilistic else 'DETERMINISTIC'}")

    try:
        # Forward cfg/setting explicitly
        logger.info("Executing: Skill Score in Bins")
        skill_score_in_bins(cfg=cfg, setting=setting)
        
        logger.info("Executing: Spatial FAR/MR/MAE Map")
        spatial_far_mr_mae_map(cfg=cfg, setting=setting)

        logger.info("MOMP Workflow completed successfully!")

    except Exception as e:
        logger.error(f"MOMP failed during execution: {e}")
        # Print full stack trace only if debug logging is enabled
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)

def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        prog="momp-run",
        description="Metrics for Onset of Monsoon Precipitation (MOMP) - Main Entry Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -p demo/et/config_et_det.in --mode prob
  %(prog)s --mode det -v
  %(prog)s -p my_config.in -q
        """
    )
    
    parser.add_argument(
        "-p", "--config",
        type=str,
        default=None,
        help="Path to the configuration file (e.g., demo/et/config_et_det.in)"
    )
    
    # MADE REQUIRED: The script will NOT run unless this is provided
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prob", "det", "probabilistic", "deterministic"],
        required=True,  # <-- This forces the user to specify it
        help="Execution mode: 'prob' (probabilistic) or 'det' (deterministic). REQUIRED."
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG level logging (shows full tracebacks on error)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Enable WARNING level logging (suppresses INFO messages)"
    )
    
    args = parser.parse_args()
    
    # 1. Configure logging dynamically
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    
    # 2. Load configuration
    if args.config:
        try:
            cfg = get_cfg(config_path=args.config)  # Adjust 'config_path=' if your loader uses positional args
        except TypeError:
            cfg = get_cfg(args.config)
    else:
        cfg = get_cfg()
        
    setting = get_setting()
    
    # 3. Apply the strictly required mode
    if args.mode in ["prob", "probabilistic"]:
        cfg.probabilistic = True
        logger.info("Execution Mode: PROBABILISTIC")
    else:
        cfg.probabilistic = False
        logger.info("Execution Mode: DETERMINISTIC")
    
    # 4. Execute
    run_momp(cfg=cfg, setting=setting)

if __name__ == "__main__":
    main()