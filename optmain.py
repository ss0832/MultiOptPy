import sys
sys.path.append('./biaspotpy')

import biaspotpy

args = biaspotpy.interface.optimizeparser()
bpa = biaspotpy.optimization.Optimize(args)
bpa.run()
