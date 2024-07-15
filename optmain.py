import sys
sys.path.append('./biaspotpy')

import biaspotpy

parser = biaspotpy.interface.init_parser()
args = biaspotpy.interface.optimizeparser(parser)
bpa = biaspotpy.optimization.Optimize(args)
bpa.run()
