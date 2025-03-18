import sys
sys.path.append('./multioptpy')

import multioptpy

parser = multioptpy.interface.init_parser()
args = multioptpy.interface.ieipparser(parser)
iEIP = multioptpy.ieip.iEIP(args)
iEIP.run()
