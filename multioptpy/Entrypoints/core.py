import multioptpy
from multioptpy.Utils.rcmc import main as rcmc_main

def run_optmain():
    """ Entry point for the main geometry optimization script (optmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.optimizeparser(parser)    
    bpa = multioptpy.optimization.Optimize(args)
    bpa.run()


def run_ieipmain():
    """ Entry point for the iEIP calculation script (ieipmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.ieipparser(parser)
    iEIP = multioptpy.ieip.iEIP(args)
    iEIP.run()


def run_mdmain():
    """ Entry point for the molecular dynamics script (mdmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.mdparser(parser)
    MD = multioptpy.moleculardynamics.MD(args)
    MD.run()


def run_nebmain():
    """ Entry point for the Nudged Elastic Band (NEB) calculation script (nebmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.nebparser(parser)
    NEB = multioptpy.neb.NEB(args)
    NEB.run()

def run_rcmc():
    """Entry point for the RCMC standalone analysis (rcmc.py).

    Usage (after ``pip install .``):
        run_rcmc reaction_network.json -T 300 -t 1.0
        run_rcmc reaction_network.json --temperature 500 --time 1e-9 \\
                 --start-node 2 --output-dir ./rcmc_out
    """
    rcmc_main()