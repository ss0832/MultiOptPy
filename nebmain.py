import sys
sys.path.append('./biaspotpy')

import biaspotpy

parser = biaspotpy.interface.init_parser()
args = biaspotpy.interface.nebparser(parser)
NEB = biaspotpy.neb.NEB(args)
NEB.run()
