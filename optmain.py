import sys
#import os
#from glob import glob
sys.path.append('./multioptpy')

import multioptpy

parser = multioptpy.interface.init_parser()
args = multioptpy.interface.optimizeparser(parser)    
bpa = multioptpy.optimization.Optimize(args)
bpa.run()
#print("----------------------------------------------------")
#print("### gradients of bias potential parameters       ###")
#print("### (This is the parameters of the final file.)  ###")
#print(bpa.bias_pot_params_grad_name_list)
#print(bpa.bias_pot_params_grad_list)
#print("----------------------------------------------------")
