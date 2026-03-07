import multioptpy


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
