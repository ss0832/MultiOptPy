import numpy as np

from multioptpy.ModelHessian.fischer import FischerApproxHessian
from multioptpy.ModelHessian.fischerd3 import FischerD3ApproxHessian
from multioptpy.ModelHessian.fischerd4 import FischerD4ApproxHessian
from multioptpy.ModelHessian.gfnff import GFNFFApproxHessian
from multioptpy.ModelHessian.gfn0xtb import GFN0XTBApproxHessian
from multioptpy.ModelHessian.lindh import LindhApproxHessian
from multioptpy.ModelHessian.lindh2007d2 import Lindh2007D2ApproxHessian
from multioptpy.ModelHessian.lindh2007d3 import Lindh2007D3ApproxHessian
from multioptpy.ModelHessian.lindh2007d4 import Lindh2007D4ApproxHessian
from multioptpy.ModelHessian.morse import MorseApproxHessian
from multioptpy.ModelHessian.schlegel import SchlegelApproxHessian
from multioptpy.ModelHessian.schlegeld3 import SchlegelD3ApproxHessian
from multioptpy.ModelHessian.schlegeld4 import SchlegelD4ApproxHessian
from multioptpy.ModelHessian.swartd2 import SwartD2ApproxHessian
from multioptpy.ModelHessian.swartd3 import SwartD3ApproxHessian
from multioptpy.ModelHessian.swartd4 import SwartD4ApproxHessian
from multioptpy.ModelHessian.shortrange import ShortRangeCorrectionHessian
from multioptpy.ModelHessian.tshess import TransitionStateHessian

from multioptpy.Parameters.parameter import UnitValueLib

class ApproxHessian:
    def __init__(self):
        return
    
    def main(self, coord, element_list, cart_gradient, approx_hess_type="lindh2007d3"):
        #coord: Bohr
        
        
        if "gfnff" in approx_hess_type.lower():
            GFNFFAH = GFNFFApproxHessian()
            hess_proj = GFNFFAH.main(coord, element_list, cart_gradient)
        elif "gfn0xtb" in approx_hess_type.lower():
            GFN0AH = GFN0XTBApproxHessian()
            hess_proj = GFN0AH.main(coord, element_list, cart_gradient)
        elif "fischerd3" in approx_hess_type.lower():
            FAHD3 = FischerD3ApproxHessian()
            hess_proj = FAHD3.main(coord, element_list, cart_gradient)
        elif "fischerd4" in approx_hess_type.lower():
            FAHD4 = FischerD4ApproxHessian()
            hess_proj = FAHD4.main(coord, element_list, cart_gradient)
        
        elif "schlegeld3" in approx_hess_type.lower():
            SAHD3 = SchlegelD3ApproxHessian()
            hess_proj = SAHD3.main(coord, element_list, cart_gradient)
        elif "schlegeld4" in approx_hess_type.lower():
            SAHD4 = SchlegelD4ApproxHessian()
            hess_proj = SAHD4.main(coord, element_list, cart_gradient)
        elif "schlegel" in approx_hess_type.lower():
            SAH = SchlegelApproxHessian()
            hess_proj = SAH.main(coord, element_list, cart_gradient)
        
        elif "swartd3" in approx_hess_type.lower():
            SWHD3 = SwartD3ApproxHessian()
            hess_proj = SWHD3.main(coord, element_list, cart_gradient)
        elif "swartd4" in approx_hess_type.lower():
            SWHD4 = SwartD4ApproxHessian()
            hess_proj = SWHD4.main(coord, element_list, cart_gradient)
        elif "swart" in approx_hess_type.lower():
            SWH = SwartD2ApproxHessian()
            hess_proj = SWH.main(coord, element_list, cart_gradient)
        elif "lindh2007d3" in approx_hess_type.lower():
            LH2007D3 = Lindh2007D3ApproxHessian()
            hess_proj = LH2007D3.main(coord, element_list, cart_gradient)
        elif "lindh2007d4" in approx_hess_type.lower():
            LH2007D4 = Lindh2007D4ApproxHessian()
            hess_proj = LH2007D4.main(coord, element_list, cart_gradient)
        elif "lindh2007" in approx_hess_type.lower():
            LH2007 = Lindh2007D2ApproxHessian()
            hess_proj = LH2007.main(coord, element_list, cart_gradient)
        elif "lindh" in approx_hess_type.lower():
            LAH = LindhApproxHessian()
            hess_proj = LAH.main(coord, element_list, cart_gradient)
        elif "fischer" in approx_hess_type.lower():
            FH = FischerApproxHessian()
            hess_proj = FH.main(coord, element_list, cart_gradient)
        elif "morse" in approx_hess_type.lower():
            MH = MorseApproxHessian()
            hess_proj = MH.create_model_hessian(coord, element_list)
        else:
            print("Approximate Hessian type not recognized. Using default Lindh (2007) D3 model...")
            LH2007D3 = Lindh2007D3ApproxHessian()
            hess_proj = LH2007D3.main(coord, element_list, cart_gradient)
            
        if "ts" in approx_hess_type.lower():
            print("Applying transition state Hessian modification...")
            TSH = TransitionStateHessian()
            hess_proj = TSH.create_ts_hessian(hess_proj, cart_gradient)
        
        if "sr" in approx_hess_type.lower():
            SRCH = ShortRangeCorrectionHessian()
            hess_proj = SRCH.main(coord, element_list, hess_proj)
        
        if "clip" in approx_hess_type.lower():
            print("Applying eigenvalue clipping...")
            #eigenvalue smoothing
            eigval, eigvec = np.linalg.eigh(hess_proj)
            eigval = np.asarray(eigval)
            eigval = smooth_eigval(eigval, alpha=0.1)
            hess_proj = np.dot(eigvec, np.dot(np.diag(eigval), eigvec.T))
          
        return hess_proj#cart_hess


def smooth_eigval(eigval, alpha=0.1):
    """Smooth eigenvalues to avoid abnormally large values"""
    eigval = np.asarray(eigval)
    result = eigval.astype(float, copy=True)
    mask = np.abs(eigval) >= 1.0
    result[mask] = np.sign(eigval[mask]) * (2.0 - 1.0 / (np.abs(eigval)[mask] ** alpha))
    return result


def test():
    AH = ApproxHessian()
    words = ["O        1.607230637      0.000000000     -4.017111134",
             "O        1.607230637      0.463701826     -2.637210910",
             "H        2.429229637      0.052572461     -2.324941515",
             "H        0.785231637     -0.516274287     -4.017735703"]
    
    elements = []
    coord = []
    
    for word in words:
        sw = word.split()
        elements.append(sw[0])
        coord.append(sw[1:4])
    
    coord = np.array(coord, dtype="float64")/UnitValueLib().bohr2angstroms#Bohr
    gradient = np.array([[-0.0028911  ,  -0.0015559   ,  0.0002471],
                         [ 0.0028769  ,  -0.0013954   ,  0.0007272],
                         [-0.0025737   ,  0.0013921   , -0.0007226],
                         [ 0.0025880   ,  0.0015592  ,  -0.0002518]], dtype="float64")#a. u.
    
    hess_proj = AH.main(coord, elements, gradient)
    
    return hess_proj



if __name__ == "__main__":#test
    test()
    
    
    