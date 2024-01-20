import sys
sys.path.append('./biaspotpy')

import biaspotpy

args = biaspotpy.interface.ieipparser()
iEIP = biaspotpy.ieip.iEIP(args)
iEIP.run()
