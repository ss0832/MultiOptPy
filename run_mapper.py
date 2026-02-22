"""
run_mapper.py - CLI entry point for the Reaction Network Mapper
===============================================================

Follows the structure of run_autots.py and adds mapper-specific
command-line arguments for temperature, RMSD threshold, etc.

Usage
-----
Basic run (config.json must exist):
    python run_mapper.py initial.xyz

Override temperature and iteration cap:
    python run_mapper.py initial.xyz --temperature 500 --max_iter 30

Override AFIR parameters from the command line (-ma mirrors run_autots.py):
    python run_mapper.py initial.xyz -ma 150.0 1,2 3,4,5

Resume an interrupted run:
    python run_mapper.py initial.xyz --resume

Use a custom config and software-path file:
    python run_mapper.py initial.xyz -cfg my_config.json -osp /path/to/software_path.conf

config.json keys understood by the mapper
-----------------------------------------
All keys used by run_autots.py (step1_settings, step2_settings,
step3_settings, step4_settings, top_n_candidates, ...) are passed
through unchanged.

An optional "mapper_settings" block controls the mapper itself:

    "mapper_settings": {
        "temperature_K"    : 300.0,    // Boltzmann temperature [K]
        "rmsd_threshold"   : 0.30,     // structure identity threshold [A]
        "max_iterations"   : 50,       // 0 = unlimited
        "afir_gamma_kJmol" : 100.0,    // AFIR gamma [kJ/mol]
        "max_pairs"        : 5,        // perturbations generated per node
        "dist_lower_ang"   : 1.5,      // distance window lower bound [A]
        "dist_upper_ang"   : 5.0,      // distance window upper bound [A]
        "output_dir"       : "mapper_output",
        "rng_seed"         : 42        // RNG seed for reproducibility
        "active_atoms"     : [1,2,5,6] // 1-based atom labels to restrict pair search
                                       // (null or omit = all atoms)
        "include_negative_gamma": false // also explore repulsive (negative gamma)
                                        // direction for each selected pair
    }

CLI arguments take precedence over mapper_settings, which in turn
takes precedence over the defaults listed above.
"""

import argparse
import json
import logging
import os
import sys


try:
    from multioptpy.Wrapper.mapper import (
        ReactionNetworkMapper,
        BoltzmannQueue,
        StructureChecker,
        PerturbationGenerator,
    )
except ImportError as _import_err:
    print(f"Error: could not import mapper.py: {_import_err}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: str = "mapper.log") -> None:
    """Configure console + file logging."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=numeric, format=fmt, datefmt=date_fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return a JSON configuration file as a dict."""
    if not os.path.isfile(path):
        print(f"Error: config file not found: {path}")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        print(f"Config loaded: {path}")
        return config
    except json.JSONDecodeError as exc:
        print(f"Error: JSON parse error in {path}: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_mapper.py",
        description="Autonomous chemical reaction network mapper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Positional (mirrors run_autots.py) ----
    parser.add_argument(
        "input_file",
        help="Path to the initial structure XYZ file.",
    )

    # ---- run_autots.py-compatible options ----
    parser.add_argument(
        "-cfg", "--config_file",
        default="./config.json",
        help="Path to the JSON configuration file. Default: ./config.json",
    )
    parser.add_argument(
        "-osp", "--software_path_file",
        default="./software_path.conf",
        help="Path to software_path.conf. Default: ./software_path.conf",
    )
    parser.add_argument(
        "-ma", "--manual_AFIR",
        nargs="*",
        required=False,
        help=(
            "Seed AFIR parameters for Step 1 (overrides step1_settings in config). "
            "Format: gamma_kJmol frag1_atoms frag2_atoms ... "
            "Example: -ma 100.0 1,2 3,4,5"
        ),
    )
    parser.add_argument(
        "-n", "--top_n",
        type=int,
        default=None,
        help="Number of NEB candidates to refine. Overrides config.",
    )

    # ---- Mapper-specific options ----
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Boltzmann temperature [K]. Default: from mapper_settings or 300.",
    )
    parser.add_argument(
        "--rmsd_threshold",
        type=float,
        default=None,
        help="RMSD threshold for structure identity [A]. Default: 0.30.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Maximum number of AutoTSWorkflow invocations (0=unlimited). Default: 50.",
    )
    parser.add_argument(
        "--afir_gamma",
        type=float,
        default=None,
        help="Auto-generated AFIR gamma [kJ/mol]. Default: 100.0.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Max AFIR atom-pair perturbations per node. Default: 5.",
    )
    parser.add_argument(
        "--dist_lower",
        type=float,
        default=None,
        help="AFIR distance window lower bound [A]. Default: 1.5.",
    )
    parser.add_argument(
        "--dist_upper",
        type=float,
        default=None,
        help="AFIR distance window upper bound [A]. Default: 5.0.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root directory for all mapper outputs. Default: mapper_output.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--active_atoms",
        nargs="+",
        type=int,
        default=None,
        metavar="ATOM_NUM",
        help=(
            "1-based atom label numbers to restrict AFIR pair search. "
            "Only pairs formed between these atoms will be generated. "
            "Example: --active_atoms 1 2 5 6  "
            "Default: all atoms are considered."
        ),
    )
    parser.add_argument(
        "--negative_gamma",
        action="store_true",
        default=False,
        help=(
            "Also generate repulsive (negative gamma) AFIR perturbations "
            "for every selected atom pair, in addition to the default "
            "attractive (positive gamma) direction."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing reaction_network.json.",
    )

    # ---- Logging ----
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    parser.add_argument(
        "--log_file",
        default="mapper.log",
        help="Log file path. Default: mapper.log.",
    )

    return parser


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------

def merge_config(args: argparse.Namespace, config: dict) -> dict:
    """
    Merge CLI arguments into config.

    Precedence (highest first):
        1. CLI arguments
        2. config["mapper_settings"]
        3. built-in defaults

    The final mapper parameters are stored under config["_mapper"] so
    that the AutoTSWorkflow keys in config are not polluted.
    """
    # ---- Standard AutoTSWorkflow keys ----
    config["initial_mol_file"] = os.path.abspath(args.input_file)
    config["software_path_file_source"] = os.path.abspath(args.software_path_file)

    local_conf = os.path.basename(args.software_path_file)
    for i in range(1, 5):
        step_key = f"step{i}_settings"
        config.setdefault(step_key, {})
        config[step_key]["software_path_file"] = local_conf

    if args.top_n is not None:
        config["top_n_candidates"] = args.top_n

    # ---- AFIR seed parameters ----
    if args.manual_AFIR is not None:
        config["step1_settings"]["manual_AFIR"] = args.manual_AFIR
        print(f"Seed manual_AFIR overridden from CLI: {args.manual_AFIR}")
    elif not config["step1_settings"].get("manual_AFIR"):
        print(
            "Info: 'manual_AFIR' not set. "
            "PerturbationGenerator will produce AFIR parameters automatically."
        )

    # ---- Mapper-specific settings ----
    ms = config.get("mapper_settings", {})

    def resolve(cli_val, json_key: str, default):
        """CLI > mapper_settings > default."""
        if cli_val is not None:
            return cli_val
        return ms.get(json_key, default)

    config["_mapper"] = {
        "temperature_K":          resolve(args.temperature,    "temperature_K",          300.0),
        "rmsd_threshold":         resolve(args.rmsd_threshold, "rmsd_threshold",         0.30),
        "max_iterations":         resolve(args.max_iter,       "max_iterations",         50),
        "afir_gamma_kJmol":       resolve(args.afir_gamma,     "afir_gamma_kJmol",       100.0),
        "max_pairs":              resolve(args.max_pairs,       "max_pairs",              5),
        "dist_lower_ang":         resolve(args.dist_lower,     "dist_lower_ang",         1.5),
        "dist_upper_ang":         resolve(args.dist_upper,     "dist_upper_ang",         5.0),
        "output_dir":             resolve(args.output_dir,     "output_dir",             "mapper_output"),
        "rng_seed":               resolve(args.rng_seed,       "rng_seed",               42),
        "resume":                 args.resume,
        # New options
        "active_atoms":           args.active_atoms if args.active_atoms is not None
                                      else ms.get("active_atoms", None),
        "include_negative_gamma": args.negative_gamma or ms.get("include_negative_gamma", False),
        # Absolute path to the loaded JSON file; forwarded to ReactionNetworkMapper
        # so a snapshot copy is saved inside output_dir at startup.
        "config_file_path":       os.path.abspath(args.config_file),
    }

    return config


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_config_summary(config: dict) -> None:
    """Print a human-readable summary of the run parameters."""
    m = config["_mapper"]
    sep = "=" * 62
    print(sep)
    print("  ReactionNetworkMapper  -  run configuration")
    print(sep)
    print(f"  Input structure : {config['initial_mol_file']}")
    print(f"  Output directory: {m['output_dir']}")
    print(f"  Max iterations  : {m['max_iterations'] or 'unlimited'}")
    print(f"  Temperature     : {m['temperature_K']} K  (Boltzmann queue)")
    print(f"  RMSD threshold  : {m['rmsd_threshold']} A")
    print(f"  AFIR gamma      : {m['afir_gamma_kJmol']} kJ/mol")
    print(f"  Max pairs/node  : {m['max_pairs']}")
    print(f"  Distance window : {m['dist_lower_ang']} - {m['dist_upper_ang']} A")
    print(f"  RNG seed        : {m['rng_seed']}")
    print(f"  Resume mode     : {'yes' if m['resume'] else 'no'}")
    active_str = ", ".join(str(a) for a in m["active_atoms"]) if m["active_atoms"] else "all"
    print(f"  Active atoms    : {active_str}")
    print(f"  Negative gamma  : {'yes' if m['include_negative_gamma'] else 'no'}")
    print(sep)


# ---------------------------------------------------------------------------
# Mapper assembly and launch
# ---------------------------------------------------------------------------

def launch_mapper(config: dict) -> None:
    """
    Assemble and run the ReactionNetworkMapper from config["_mapper"].

    HOW TO SWAP IN A CUSTOM STRATEGY
    ---------------------------------
    Each component (queue, checker, perturber) is constructed
    independently here before being passed to the mapper.
    To use a different priority strategy:

        1. Subclass ExplorationQueue in mapper.py (or a separate file).
        2. Instantiate your subclass here instead of BoltzmannQueue.
        3. Pass it as queue=<your_instance> to ReactionNetworkMapper.

    Example:
        from mapper import ExplorationQueue, ExplorationTask

        class BarrierFirstQueue(ExplorationQueue):
            def compute_priority(self, task):
                return -task.metadata.get("barrier_kcalmol", 0.0)
            def should_add(self, node, ref_e, **kw):
                return True

        queue = BarrierFirstQueue()
        # ... build mapper with queue=queue
    """
    m = config["_mapper"]

    queue = BoltzmannQueue(
        temperature_K=m["temperature_K"],
        rng_seed=m["rng_seed"],
    )
    checker = StructureChecker(
        rmsd_threshold=m["rmsd_threshold"],
    )
    perturber = PerturbationGenerator(
        afir_gamma_kJmol=m["afir_gamma_kJmol"],
        max_pairs=m["max_pairs"],
        dist_lower_ang=m["dist_lower_ang"],
        dist_upper_ang=m["dist_upper_ang"],
        rng_seed=m["rng_seed"],
        active_atoms=m["active_atoms"],
        include_negative_gamma=m["include_negative_gamma"],
    )

    mapper = ReactionNetworkMapper(
        base_config=config,
        queue=queue,
        structure_checker=checker,
        perturbation_generator=perturber,
        output_dir=m["output_dir"],
        graph_json="reaction_network.json",
        max_iterations=m["max_iterations"],
        resume=m["resume"],
        # config_file_path is stored in _mapper by merge_config; forwarding it
        # here causes ReactionNetworkMapper to copy the JSON to output_dir as
        # config_snapshot.json so the run is fully self-contained.
        config_file_path=m.get("config_file_path"),
    )
    mapper.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file)
    log = logging.getLogger(__name__)

    if not os.path.isfile(args.input_file):
        log.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    config = load_config(args.config_file)
    config = merge_config(args, config)
    print_config_summary(config)

    log.info("Configuration merged. Starting mapper.")
    launch_mapper(config)


if __name__ == "__main__":
    main()