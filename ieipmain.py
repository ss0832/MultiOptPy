import sys
sys.path.append('./biaspotpy')

import biaspotpy

parser = biaspotpy.interface.init_parser()
args = biaspotpy.interface.ieipparser(parser)
iEIP = biaspotpy.ieip.iEIP(args)
iEIP.run()
