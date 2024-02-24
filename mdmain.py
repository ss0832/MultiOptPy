import sys
sys.path.append('./biaspotpy')

import biaspotpy

args = biaspotpy.interface.mdparser()
MD = biaspotpy.moleculardynamics.MD(args)
MD.run()
