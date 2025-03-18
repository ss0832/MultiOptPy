import sys
sys.path.append('./multioptpy')

import multioptpy

parser = multioptpy.interface.init_parser()
args = multioptpy.interface.mdparser(parser)
MD = multioptpy.moleculardynamics.MD(args)
MD.run()
