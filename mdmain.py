import sys
sys.path.append('./biaspotpy')

import biaspotpy

parser = biaspotpy.interface.init_parser()
args = biaspotpy.interface.mdparser(parser)
MD = biaspotpy.moleculardynamics.MD(args)
MD.run()
