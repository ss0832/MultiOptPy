import sys
sys.path.append('./biaspotpy')

import biaspotpy

args = biaspotpy.interface.optimizeparser()
bpa = biaspotpy.interface.Optimize(args)
bpa.run()
