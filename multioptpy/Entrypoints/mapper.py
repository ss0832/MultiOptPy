import sys
import os
import json
import argparse
import logging


def run_mapper():
    """Entry point for the Reaction Network Mapper (mapper.py).

    Autonomously maps a chemical reaction network by iteratively calling
    AutoTSWorkflow and exploring new equilibrium structures via AFIR
    perturbations, using a Boltzmann-weighted (or RCMC) priority queue.

    Usage (after pip install)
    -------------------------
    run_mapper initial.xyz
    run_mapper initial.xyz --temperature 500 --max_iter 30
    run_mapper initial.xyz -ma 150.0 1,2 3,4,5
    run_mapper initial.xyz --active_atoms 1 2 5 6 --negative_gamma
    run_mapper initial.xyz --resume
    run_mapper initial.xyz -cfg my_config.json -osp /path/to/software_path.conf
    run_mapper initial.xyz --exclude_nodes 3 7 12
    run_mapper initial.xyz --exclude_bond_rearrangement
    run_mapper initial.xyz --use_rcmc --rcmc_temperature 500 --rcmc_time 1e-6

    config.json  --  mapper_settings block
    ---------------------------------------
    An optional "mapper_settings" object inside config.json controls the
    mapper engine. CLI arguments take precedence over this block, which in
    turn takes precedence over the built-in defaults shown below:

        "mapper_settings": {
            "temperature_K"              : 300.0,
            "rmsd_threshold"             : 0.30,
            "max_iterations"             : 50,
            "afir_gamma_kJmol"           : 100.0,
            "max_pairs"                  : 5,
            "dist_lower_ang"             : 1.5,
            "dist_upper_ang"             : 5.0,
            "output_dir"                 : "mapper_output",
            "rng_seed"                   : 42,
            "active_atoms"               : null,
            "include_negative_gamma"     : false,
            "excluded_node_ids"          : [],
            "exclude_bond_rearrangement" : false,
            "use_rcmc"                   : false,
            "rcmc_temperature_K"         : 300.0,
            "rcmc_reaction_time_s"       : 1.0
        }
    """

    # ------------------------------------------------------------------
    # Lazy import of mapper components (keeps top-level import clean)
    # ------------------------------------------------------------------
    try:
        from multioptpy.Wrapper.mapper import (
            ReactionNetworkMapper,
            BoltzmannQueue,
            StructureChecker,
            PerturbationGenerator,
            BondTopologyChecker,
        )
    except ImportError as _err:
        print(f"Error: could not import mapper components: {_err}")
        sys.exit(1)

    try:
        from multioptpy.Utils.rcmc import RCMCQueue
    except ImportError as _rcmc_err:
        print(f"Warning: could not import RCMCQueue: {_rcmc_err}")
        RCMCQueue = None

    # ==================================================================
    # Nested helpers (mirrors the structure of run_autots())
    # ==================================================================

    def setup_logging(level="INFO", log_file="mapper.log"):
        """Configure console + file logging."""
        numeric = getattr(logging, level.upper(), logging.INFO)
        fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ]
        logging.basicConfig(
            level=numeric, format=fmt, datefmt=date_fmt, handlers=handlers
        )

    def load_config(path):
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

    def build_parser():
        parser = argparse.ArgumentParser(
            prog="run_mapper",
            description="Autonomous chemical reaction network mapper.",
        )

        # ---- Positional ----
        parser.add_argument(
            "input_file",
            help="Path to the initial structure XYZ file.",
        )

        # ---- run_autots-compatible options ----
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
            "--exclude_nodes",
            nargs="+",
            type=int,
            default=None,
            metavar="NODE_ID",
            help=(
                "EQ node IDs to exclude from AFIR exploration (2nd run onwards). "
                "Example: --exclude_nodes 3 7 12  Default: no exclusions."
            ),
        )
        parser.add_argument(
            "--exclude_bond_rearrangement",
            action="store_true",
            default=False,
            help=(
                "Automatically exclude any newly found EQ node whose covalent "
                "bond topology differs from that of the seed structure (EQ0). "
                "Default: disabled."
            ),
        )
        parser.add_argument(
            "--use_rcmc",
            action="store_true",
            default=False,
            help=(
                "Use the RCMC algorithm for exploration priorities instead of "
                "the default Boltzmann queue. Default: disabled."
            ),
        )
        parser.add_argument(
            "--rcmc_temperature",
            type=float,
            default=None,
            help="Temperature [K] for the RCMC priority calculation. Default: 300.",
        )
        parser.add_argument(
            "--rcmc_time",
            type=float,
            default=None,
            help="Reaction time [s] for the RCMC priority calculation. Default: 1.0.",
        )
        parser.add_argument(
            "--rcmc_start_node",
            type=int,
            default=None,
            help=(
                "EQ node ID to use as the initial population source in the RCMC "
                "kinetic simulation.  The transient population vector is initialised "
                "with p[start_node]=1 before contraction.  "
                "Only used when --use_rcmc is set.  Default: from "
                "mapper_settings[\"rcmc_start_node_id\"] or 0 (EQ0)."
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

    def merge_config(args, config):
        """Merge CLI arguments into config.

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

        def resolve(cli_val, json_key, default):
            """CLI > mapper_settings > default."""
            if cli_val is not None:
                return cli_val
            return ms.get(json_key, default)

        config["_mapper"] = {
            "temperature_K":               resolve(args.temperature,    "temperature_K",          300.0),
            "rmsd_threshold":              resolve(args.rmsd_threshold, "rmsd_threshold",         0.30),
            "max_iterations":              resolve(args.max_iter,       "max_iterations",         50),
            "afir_gamma_kJmol":            resolve(args.afir_gamma,     "afir_gamma_kJmol",       100.0),
            "max_pairs":                   resolve(args.max_pairs,       "max_pairs",              5),
            "dist_lower_ang":              resolve(args.dist_lower,     "dist_lower_ang",         1.5),
            "dist_upper_ang":              resolve(args.dist_upper,     "dist_upper_ang",         5.0),
            "output_dir":                  resolve(args.output_dir,     "output_dir",             "mapper_output"),
            "rng_seed":                    resolve(args.rng_seed,       "rng_seed",               42),
            "resume":                      args.resume,
            # Atom-pair restrictions
            "active_atoms":                args.active_atoms if args.active_atoms is not None
                                               else ms.get("active_atoms", None),
            "include_negative_gamma":      args.negative_gamma or ms.get("include_negative_gamma", False),
            # EQ exclusion options
            "excluded_node_ids":           (
                                               list(args.exclude_nodes) if args.exclude_nodes is not None
                                               else ms.get("excluded_node_ids", None)
                                           ),
            "exclude_bond_rearrangement":  args.exclude_bond_rearrangement or ms.get("exclude_bond_rearrangement", False),
            # RCMC options
            "use_rcmc":                    args.use_rcmc or ms.get("use_rcmc", False),
            "rcmc_temperature_K":          resolve(args.rcmc_temperature, "rcmc_temperature_K",   300.0),
            "rcmc_reaction_time_s":        resolve(args.rcmc_time,        "rcmc_reaction_time_s", 1.0),
            "rcmc_start_node_id":          resolve(args.rcmc_start_node,  "rcmc_start_node_id",   0),
            # Config snapshot
            "config_file_path":            os.path.abspath(args.config_file),
        }

        return config

    def print_config_summary(config):
        """Print a human-readable summary of the run parameters."""
        m = config["_mapper"]
        sep = "=" * 62
        print(sep)
        print("  ReactionNetworkMapper  -  run configuration")
        print(sep)
        print(f"  Input structure : {config['initial_mol_file']}")
        print(f"  Output directory: {m['output_dir']}")
        print(f"  Max iterations  : {m['max_iterations'] or 'unlimited'}")
        print(f"  RMSD threshold  : {m['rmsd_threshold']} A")
        print(f"  AFIR gamma      : {m['afir_gamma_kJmol']} kJ/mol")
        print(f"  Max pairs/node  : {m['max_pairs']}")
        print(f"  Distance window : {m['dist_lower_ang']} - {m['dist_upper_ang']} A")
        print(f"  RNG seed        : {m['rng_seed']}")
        print(f"  Resume mode     : {'yes' if m['resume'] else 'no'}")
        active_str = ", ".join(str(a) for a in m["active_atoms"]) if m["active_atoms"] else "all"
        print(f"  Active atoms    : {active_str}")
        print(f"  Negative gamma  : {'yes' if m['include_negative_gamma'] else 'no'}")
        excl_ids = m.get("excluded_node_ids")
        excl_str = ", ".join(str(n) for n in sorted(excl_ids)) if excl_ids else "none"
        print(f"  Excluded EQ IDs : {excl_str}")
        print(f"  Excl. bond rearr: {'yes' if m.get('exclude_bond_rearrangement') else 'no'}")
        if m.get("use_rcmc"):
            print(f"  Priority queue  : RCMC  "
                  f"T={m['rcmc_temperature_K']} K  "
                  f"t={m['rcmc_reaction_time_s']} s  "
                  f"start_node=EQ{m['rcmc_start_node_id']}")
        else:
            print(f"  Priority queue  : Boltzmann  T={m['temperature_K']} K")
        print(sep)

    def launch_mapper(config):
        """Assemble and run the ReactionNetworkMapper from config["_mapper"].

        HOW TO SWAP IN A CUSTOM STRATEGY
        ---------------------------------
        Each component (queue, checker, perturber) is constructed
        independently here before being passed to the mapper.
        To use a different priority strategy:

            1. Subclass ExplorationQueue in mapper.py (or a separate file).
            2. Instantiate your subclass here instead of BoltzmannQueue.
            3. Pass it as queue=<your_instance> to ReactionNetworkMapper.

        The config_file_path required for the snapshot feature is read
        directly from config["_mapper"]["config_file_path"], which is
        populated by merge_config() from args.config_file.
        """
        m = config["_mapper"]

        # ── Priority queue ────────────────────────────────────────────────
        if m.get("use_rcmc", False):
            if RCMCQueue is None:
                print(
                    "Error: --use_rcmc requested but RCMCQueue could not be imported. "
                    "Check that multioptpy.Utils.rcmc is available."
                )
                sys.exit(1)
            queue = RCMCQueue(
                temperature_K=m["rcmc_temperature_K"],
                reaction_time_s=m["rcmc_reaction_time_s"],
                rng_seed=m["rng_seed"],
                start_node_id=m["rcmc_start_node_id"],
                output_dir=m["output_dir"],
            )
        else:
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
            config_file_path=m.get("config_file_path"),
            excluded_node_ids=m.get("excluded_node_ids"),
            exclude_bond_rearrangement=m.get("exclude_bond_rearrangement", False),
        )
        mapper.run()

    # ==================================================================
    # main() -- mirrors run_autots() pattern
    # ==================================================================

    def main():
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

    main()
