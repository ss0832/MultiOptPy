import sys
sys.path.append('./multioptpy')

import multioptpy

parser = multioptpy.interface.init_parser()
args = multioptpy.interface.nebparser(parser)
NEB = multioptpy.neb.NEB(args)
NEB.run()
