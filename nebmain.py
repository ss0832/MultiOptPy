import sys
sys.path.append('./biaspotpy')

import biaspotpy

args = biaspotpy.interface.nebparser()
NEB = biaspotpy.neb.NEB(args)
NEB.run()
