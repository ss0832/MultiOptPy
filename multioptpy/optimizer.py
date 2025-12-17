import numpy as np
import copy

from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass
from multioptpy.Utils.calc_tools import Calculationtools

from multioptpy.Optimizer.adabelief import Adabelief
#from multioptpy.Optimizer.fastadabelief import FastAdabelief
#from multioptpy.Optimizer.adaderivative import Adaderivative
#from multioptpy.Optimizer.sadam import SAdam
#from multioptpy.Optimizer.samsgrad import SAMSGrad
#from multioptpy.Optimizer.QHAdam import QHAdam
#from multioptpy.Optimizer.adamax import AdaMax
#from multioptpy.Optimizer.yogi import YOGI
#from multioptpy.Optimizer.nadam import NAdam
from multioptpy.Optimizer.fire import FIRE
from multioptpy.Optimizer.abc_fire import ABC_FIRE
from multioptpy.Optimizer.fire2 import FIRE2
#from multioptpy.Optimizer.adadiff import AdaDiff
#from multioptpy.Optimizer.adamod import Adamod
from multioptpy.Optimizer.radam import RADAM
from multioptpy.Optimizer.eve import EVE
#from multioptpy.Optimizer.adamw import AdamW
from multioptpy.Optimizer.adam import Adam
#from multioptpy.Optimizer.adafactor import Adafactor
from multioptpy.Optimizer.prodigy import Prodigy
#from multioptpy.Optimizer.adabound import AdaBound
#from multioptpy.Optimizer.adadelta import Adadelta
from multioptpy.Optimizer.conjugate_gradient import ConjgateGradient
#from multioptpy.Optimizer.hybrid_rfo import HybridRFO
#from multioptpy.Optimizer.rfo import RationalFunctionOptimization
#from multioptpy.Optimizer.ric_rfo import RedundantInternalRFO
from multioptpy.Optimizer.rsprfo import EnhancedRSPRFO
from multioptpy.Optimizer.rsirfo import RSIRFO
from multioptpy.Optimizer.crsirfo import CRSIRFO
from multioptpy.Optimizer.mf_rsirfo import MF_RSIRFO
#from multioptpy.Optimizer.newton import Newton
from multioptpy.Optimizer.lbfgs import LBFGS
from multioptpy.Optimizer.tr_lbfgs import TRLBFGS
#from multioptpy.Optimizer.rmspropgrave import RMSpropGrave
from multioptpy.Optimizer.lookahead import LookAhead
from multioptpy.Optimizer.lars import LARS
from multioptpy.Optimizer.gdiis import GDIIS
from multioptpy.Optimizer.ediis import EDIIS
from multioptpy.Optimizer.gediis import GEDIIS
from multioptpy.Optimizer.c2diis import C2DIIS
from multioptpy.Optimizer.adiis import ADIIS 
from multioptpy.Optimizer.kdiis import KrylovDIIS as KDIIS
from multioptpy.Optimizer.gpr_step import GPRStep
from multioptpy.Optimizer.gan_step import GANStep
from multioptpy.Optimizer.rl_step import RLStepSizeOptimizer
from multioptpy.Optimizer.geodesic_step import GeodesicStepper
from multioptpy.Optimizer.linesearch import LineSearch
from multioptpy.Optimizer.component_wise_scaling import ComponentWiseScaling
from multioptpy.Optimizer.coordinate_locking import CoordinateLocking
from multioptpy.Optimizer.trust_radius import TrustRadius
from multioptpy.Optimizer.gradientdescent import GradientDescent, MassWeightedGradientDescent
from multioptpy.Optimizer.gpmin import GPmin
from multioptpy.Optimizer.trim import TRIM

optimizer_mapping = {
    "adabelief": Adabelief,
    "radam": RADAM,
    "eve": EVE,
    "prodigy": Prodigy,
    "abcfire": ABC_FIRE,
    "fire2": FIRE2,
    "fire": FIRE,
    "mwgradientdescent": MassWeightedGradientDescent,
    "gradientdescent": GradientDescent,
    "steepest_descent": GradientDescent,
    "gpmin": GPmin,
    "tr_lbfgs": TRLBFGS,
    "lbfgs": LBFGS,
}

specific_cases = {
    "ranger": {"optimizer": RADAM, "lookahead": LookAhead(), "lars": None},
    "rangerlars": {"optimizer": RADAM, "lookahead": LookAhead(), "lars": LARS()},
    "adam": {"optimizer": Adam, "lookahead": LookAhead(), "lars": None},
    "adamlars": {"optimizer": Adam, "lookahead": None, "lars": LARS()},
    "adamlookahead": {"optimizer": Adam, "lookahead": LookAhead(), "lars": None},
    "adamlookaheadlars": {"optimizer": Adam, "lookahead": LookAhead(), "lars": LARS()},
}

quasi_newton_mapping = { 
    "mwsmf_rsirfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "mwsmf_rsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},


    "mwmf_rsirfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "mwmf_rsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},

    "smf_rsirfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "smf_rsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},


    "mf_rsirfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "mf_rsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},
   
    "crsirfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "crsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},
   
   
    "rsirfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},

   
  
    "rsprfo_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_bfgs_dd": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_cfd_fsb_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_cfd_fsb_dd": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_cfd_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_cfd_bofill_weighted": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_block_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_cfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_pcfd_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_msp": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_psb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_flowchart": {"delta": 0.50, "rfo_type": 1},   
}



class CalculateMoveVector:
    def __init__(self, DELTA, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0, model_hess_flag=None, max_trust_radius=None, min_trust_radius=None, **kwargs):
        self.DELTA = DELTA
        self.temperature = temperature
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.unitval = UnitValueLib()
        self.FC_COUNT = FC_COUNT
        self.MAX_MAX_FORCE_SWITCHING_THRESHOLD = 0.0050
        self.MIN_MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.MAX_RMS_FORCE_SWITCHING_THRESHOLD = 0.05
        self.MIN_RMS_FORCE_SWITCHING_THRESHOLD = 0.005
        self.projection_constraint = kwargs.get("projection_constraint", None) 
        self.max_trust_radius = max_trust_radius        
        self.min_trust_radius = min_trust_radius
        self.CALC_TRUST_RADII = TrustRadius()
        if self.max_trust_radius is not None:
            if self.max_trust_radius <= 0.0:
                print("max_trust_radius must be greater than 0.0")
                exit()
                
            self.CALC_TRUST_RADII.set_max_trust_radius(self.max_trust_radius)
            
        if self.max_trust_radius is None:
            if saddle_order > 0:
                self.max_trust_radius = 0.1
                self.trust_radii = 0.1
            else:
                self.max_trust_radius = 0.5
                self.trust_radii = 0.5
        else:
            if saddle_order > 0:
                self.trust_radii = min(self.max_trust_radius, 0.1)
            else:
                self.trust_radii = self.max_trust_radius if type(self.max_trust_radius) is float else 0.5

        if self.min_trust_radius is not None:
            if self.min_trust_radius <= 0.0:
                print("min_trust_radius must be greater than 0.0")
                exit()
            if self.trust_radii < self.min_trust_radius:
                self.trust_radius = self.min_trust_radius
                
            self.CALC_TRUST_RADII.set_min_trust_radius(self.min_trust_radius)

        self.min_trust_radius = min_trust_radius if min_trust_radius is not None else 0.01

        self.saddle_order = saddle_order
        self.iter = 0
        self.element_list = element_list
        self.model_hess_flag = model_hess_flag

    def initialization(self, method):
        """
        Initializes the optimizer instances based on the provided method names.
        
        This function parses a list of method strings. Each string defines a
        base optimizer (e.g., LBFGS, FIRE, RSIRFO) and optionally
        a set of enhancements (e.g., "lookahead", "gdiis", "lars")
        which are chained together.
        
        Args:
            method (list[str]): A list of method name strings.
        
        Returns:
            list: A list of initialized base optimizer instances.
        """

        # --- Helper Function to Handle Enhancements ---
        
        def _append_enhancements(lower_m, is_newton_method, specific_lookahead=None, specific_lars=None):
            """
            Private helper to append all enhancement instances for a given method.
            This avoids duplicating this logic in every 'if/elif' block.
            
            Args:
                lower_m (str): The lowercased method name string.
                is_newton_method (bool): Flag indicating if the base optimizer is a quasi-Newton method.
                specific_lookahead (object, optional): A pre-defined LookAhead instance from 'specific_cases'.
                specific_lars (object, optional): A pre-defined LARS instance from 'specific_cases'.
            """
            
            # Handle LookAhead and LARS, prioritizing 'specific_cases' config
            if specific_lookahead is not None:
                lookahead_instances.append(specific_lookahead)
            else:
                lookahead_instances.append(LookAhead() if "lookahead" in lower_m else None)

            if specific_lars is not None:
                lars_instances.append(specific_lars)
            else:
                lars_instances.append(LARS() if "lars" in lower_m else None)
            
            # LineSearch
            linesearch_instances.append(LineSearch() if "linesearch" in lower_m else None)
            
            
            # DIIS family
            gdiis_instances.append(GDIIS() if "gdiis" in lower_m else None)
            kdiis_instances.append(KDIIS() if "kdiis" in lower_m else None)
            adiis_instances.append(ADIIS() if "adiis" in lower_m else None)
            c2diis_instances.append(C2DIIS() if "c2diis" in lower_m else None)
            
            
            # Handle mutually exclusive EDIIS/GEDIIS
            if "gediis" in lower_m:
                gediis_instances.append(GEDIIS())
                ediis_instances.append(None)
            else:
                ediis_instances.append(EDIIS() if "ediis" in lower_m else None)
                gediis_instances.append(None)
                
            # Coordinate transformations
            coordinate_locking_instances.append(CoordinateLocking() if "coordinate_locking" in lower_m else None)
            coordinate_wise_scaling_instances.append(ComponentWiseScaling() if "component_wise_scaling" in lower_m else None)
            
            # ML-based step optimizers
            gpr_step_instances.append(GPRStep() if "gpr_step" in lower_m else None)
            gan_step_instances.append(GANStep() if "gan_step" in lower_m else None)
            rl_step_instances.append(RLStepSizeOptimizer() if "rl_step" in lower_m else None)
            
            # Other step modifiers
            geodesic_step_instances.append(GeodesicStepper(element_list=self.element_list) if "geodesic_step" in lower_m else None)
            
            # TRIM is only relevant for quasi-Newton methods
            if is_newton_method:
                trim_step_instances.append(TRIM(saddle_order=self.saddle_order) if "trim" in lower_m else None)
            else:
                trim_step_instances.append(None)

        # --- End of Helper Function ---

        # Initialize lists to store instances for each method
        optimizer_instances = []
        newton_tag = []
        lookahead_instances = []
        lars_instances = []
        gdiis_instances = []
        ediis_instances = []
        gediis_instances = []
        c2diis_instances = []
        adiis_instances = []
        kdiis_instances = []
        coordinate_locking_instances = []
        coordinate_wise_scaling_instances = []
        gpr_step_instances = []
        gan_step_instances = []
        rl_step_instances = []
        geodesic_step_instances = []
        trim_step_instances = []
        linesearch_instances = []
        
        # Loop over each requested method string
        for i, m in enumerate(method):
            lower_m = m.lower()
            optimizer_added = False
            is_newton = False
            optimizer = None

            # 1. Check hard-coded specific cases (e.g., "ranger")
            if lower_m in specific_cases:
                case = specific_cases[lower_m]
                optimizer = case["optimizer"]()
                is_newton = False
                optimizer_instances.append(optimizer)
                newton_tag.append(is_newton)
                _append_enhancements(lower_m, 
                                     is_newton, 
                                     specific_lookahead=case.get("lookahead"), 
                                     specific_lars=case.get("lars"))
                optimizer_added = True

            # 2. Check quasi-Newton methods
            elif any(key in lower_m for key in quasi_newton_mapping):
                for key, settings in quasi_newton_mapping.items():
                    if key in lower_m:
                        print(key)
                        if "rsprfo" in key:
                            optimizer = EnhancedRSPRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius)
                        
                        elif "mwsmf_rsirfo" in key:
                            optimizer = MF_RSIRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius, adaptive_mode_following=False)
                        
                        elif "mwmf_rsirfo" in key:
                            optimizer = MF_RSIRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius)
                        
                        elif "smf_rsirfo" in key:
                            optimizer = MF_RSIRFO(method=m, saddle_order=self.saddle_order, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius, adaptive_mode_following=False)
                        
                        elif "mf_rsirfo" in key:
                            optimizer = MF_RSIRFO(method=m, saddle_order=self.saddle_order, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius)
                        
                        elif "crsirfo" in key and self.projection_constraint:
                            optimizer = CRSIRFO(method=m, constraints=self.projection_constraint, saddle_order=self.saddle_order, element_list=self.element_list, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius)                        
                        elif "rsirfo" in key:
                            optimizer = RSIRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list, trust_radius_max=self.max_trust_radius, trust_radius_min=self.min_trust_radius)
                        else:
                            print(f"This method is not implemented: {m}. Thus, exiting.")
                            exit()

                        optimizer.DELTA = settings["delta"]
                        if "linesearch" in settings:
                            optimizer.linesearchflag = True
                        
                        is_newton = True
                        optimizer_instances.append(optimizer)
                        newton_tag.append(is_newton)
                        _append_enhancements(lower_m, is_newton)
                        optimizer_added = True
                        break  # Exit inner loop once match is found
            
            # 3. Check standard gradient-based optimizers
            if not optimizer_added:
                for key, optimizer_class in optimizer_mapping.items():
                    if key in lower_m:
                        optimizer = optimizer_class()
                        if lower_m == "mwgradientdescent":
                            optimizer.element_list = self.element_list
                            optimizer.atomic_mass = atomic_mass
                        
                        is_newton = False
                        optimizer_instances.append(optimizer)
                        newton_tag.append(is_newton)
                        _append_enhancements(lower_m, is_newton)
                        optimizer_added = True
                        break  # Exit inner loop once match is found
            
            # 4. Check Conjugate Gradient methods
            # Define CG keys. Put "cg" last so "cg_pr" matches first if present.
            cg_keys = ["cg_pr", "cg_fr", "cg_hs", "cg_dy", "cg"] 
            if not optimizer_added:
                for key in cg_keys:
                    if key in lower_m:
                        # Use the matched key (e.g., "cg" or "cg_pr") for the constructor,
                        optimizer = ConjgateGradient(method=key) 
                        is_newton = False
                        optimizer_instances.append(optimizer)
                        newton_tag.append(is_newton)
                        # Pass the full 'lower_m' string to check for other enhancements
                        _append_enhancements(lower_m, is_newton) 
                        optimizer_added = True
                        break # Found a cg match, exit inner loop

            # 5. Handle default case (method not found)
            if not optimizer_added:
                print(f"This method is not implemented: {m}. Thus, Default method (FIRE) is used.")
                optimizer = FIRE()
                is_newton = False
                optimizer_instances.append(optimizer)
                newton_tag.append(is_newton)
                _append_enhancements(lower_m, is_newton)
        
        # --- End of loop ---

        # Store all instance lists as class attributes
        # These are used by other methods in the class
        self.method = method
        self.newton_tag = newton_tag
        self.lookahead_instances = lookahead_instances
        self.lars_instances = lars_instances
        self.gdiis_instances = gdiis_instances
        self.ediis_instances = ediis_instances
        self.gediis_instances = gediis_instances
        self.c2diis_instances = c2diis_instances
        self.adiis_instances = adiis_instances
        self.kdiis_instances = kdiis_instances
        self.coordinate_locking_instances = coordinate_locking_instances
        self.coordinate_wise_scaling_instances = coordinate_wise_scaling_instances
        self.gpr_step_instances = gpr_step_instances
        self.gan_step_instances = gan_step_instances
        self.rl_step_instances = rl_step_instances
        self.geodesic_step_instances = geodesic_step_instances
        self.trim_step_instances = trim_step_instances
        self.linesearch_instances = linesearch_instances
        return optimizer_instances  

    def update_trust_radius_conditionally(self, optimizer_instances, B_e, pre_B_e, pre_B_g, pre_move_vector, geom_num_list):
        """
        Refactored method to handle trust radius updates and conditions.
        """
        # Early exit if no full-coordinate count and no Hessian flag
        if self.FC_COUNT == -1 and not self.model_hess_flag is not None:
            return

        # Determine if there's a model Hessian to use
        model_hess = None
        for i in range(len(optimizer_instances)):
            if self.newton_tag[i]:
                model_hess = optimizer_instances[i].hessian + optimizer_instances[i].bias_hessian
                break

        # Update trust radii only if we have a Hessian or if FC_COUNT is not -1
        if not (model_hess is None and self.FC_COUNT == -1) and True in self.newton_tag:
            self.trust_radii = self.CALC_TRUST_RADII.update_trust_radii(
                B_e, pre_B_e, pre_B_g, pre_move_vector, model_hess, geom_num_list, self.trust_radii
            )

        if self.min_trust_radius is not None:
            print("user_difined_min_trust_radius: ", self.min_trust_radius)

        if self.max_trust_radius is not None:
            print("user_difined_max_trust_radius: ", self.max_trust_radius)
        else:
            # If saddle order is positive, constrain the trust radii
            if self.saddle_order > 0:
                self.trust_radii = min(self.trust_radii, self.max_trust_radius if self.max_trust_radius is not None else 0.1)

            # If this is the first iteration but not full-coordinate -1 check
            if self.iter == 0 and self.FC_COUNT != -1:
                if self.saddle_order > 0:
                    self.trust_radii = min(self.trust_radii, self.max_trust_radius if self.max_trust_radius is not None else 0.1)

    def handle_projection_constraint(self, projection_constrain):
        """
        Constrain the trust radii if projection constraint is enabled.
        """
        if projection_constrain:
            if self.max_trust_radius is not None:
                pass
            else:
                self.trust_radii = min(self.trust_radii, self.max_trust_radius if self.max_trust_radius is not None else 0.1)



    def switch_move_vector(self,
        B_g,
        move_vector_list,
        method_list=None,
        max_rms_force_switching_threshold=0.05,
        min_rms_force_switching_threshold=0.005,
        steepness=10.0,
        offset=0.5):
        

        if len(method_list) == 1:
            return copy.copy(move_vector_list[0]), method_list

        rms_force = abs(np.sqrt(np.square(B_g).mean()))

        if rms_force > max_rms_force_switching_threshold:

            print(f"Switching to {method_list[0]}")
            return copy.copy(move_vector_list[0]), method_list
        elif min_rms_force_switching_threshold < rms_force <= max_rms_force_switching_threshold:

            x_j = (rms_force - min_rms_force_switching_threshold) / (
                max_rms_force_switching_threshold - min_rms_force_switching_threshold
            )

            f_val = 1 / (1 + np.exp(-steepness * (x_j - offset)))
            combined_vector = (
                np.array(move_vector_list[0], dtype="float64") * f_val
                + np.array(move_vector_list[1], dtype="float64") * (1.0 - f_val)
            )
            print(f"Weighted switching: {f_val:.3f} {method_list[0]} {method_list[1]}")
            return combined_vector, method_list
        else:

            print(f"Switching to {method_list[1]}")
            return copy.copy(move_vector_list[1]), method_list
        
    def update_move_vector_list(
            self,
            optimizer_instances,
            B_g,
            pre_B_g,
            pre_geom,
            B_e,
            pre_B_e,
            pre_move_vector,
            initial_geom_num_list,
            g,
            pre_g,
        ):
        """
        Update a list of move vectors from multiple optimizer instances,
        including optional enhancements through various techniques.
        """
        move_vector_list = []
      
        tmp_hess = None
        for i in range(len(optimizer_instances)):
            if self.newton_tag[i]:
                tmp_hess = optimizer_instances[i].hessian + optimizer_instances[i].bias_hessian
                break
        # Enhancement techniques and their parameter configurations
        # Format: [list_of_instances, method_name, [parameters]]
        enhancement_config = [
            # Base optimizer - separate handling
            
            # Trust region and momentum techniques
            [self.lars_instances, "apply_lars", [B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, 
                                                initial_geom_num_list, g, pre_g]],
            [self.lookahead_instances, "apply_lookahead", [B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector,
                                                        initial_geom_num_list, g, pre_g]],
            
            # linesearch
            [self.linesearch_instances, "apply_linesearch", [B_g, pre_B_g, B_e, pre_B_e]],
            
            # DIIS family techniques
            [self.ediis_instances, "apply_ediis", [B_e, B_g]],
            [self.gdiis_instances, "apply_gdiis", [B_g, pre_B_g]],
            [self.c2diis_instances, "apply_c2diis", [B_g, pre_B_g]],
            [self.adiis_instances, "apply_adiis", [B_e, B_g]],
            [self.kdiis_instances, "apply_kdiis", [B_e, B_g]],
            [self.gediis_instances, "apply_gediis", [B_e, B_g, pre_B_g]],
            
            # Coordinate transformation techniques
            [self.coordinate_locking_instances, "apply_coordinate_locking", [B_e, B_g]],
            [self.coordinate_wise_scaling_instances, "apply_coordinate_scaling", [B_e, B_g]],
            
            # ML-based step prediction techniques
            [self.gpr_step_instances, "apply_ml_step", [B_e, B_g]],
            
            # GAN-based step prediction techniques
            [self.gan_step_instances, "apply_gan_step", [B_e, B_g]],
            
            # Reinforcement learning-based step prediction techniques
            [self.rl_step_instances, "apply_rl_step", [B_g, pre_B_g, B_e, pre_B_e]],
            # Geodesic stepping techniques
            [self.geodesic_step_instances, "apply_geodesic_step", []],
            # TRIM step adjustment techniques
            [self.trim_step_instances, "apply_trim_step", [B_g, tmp_hess, self.trust_radii]],
            ]

        for i, optimizer_instance in enumerate(optimizer_instances):
            # Get initial move vector from base optimizer
            tmp_move_vector = optimizer_instance.run(
                self.geom_num_list,
                B_g,
                pre_B_g,
                pre_geom,
                B_e,
                pre_B_e,
                pre_move_vector,
                initial_geom_num_list,
                g,
                pre_g
            )
            tmp_move_vector = np.array(tmp_move_vector, dtype="float64")

            # Apply each enhancement technique if available
            for instance_list, method_name, base_params in enhancement_config:
                if i < len(instance_list) and instance_list[i] is not None:
                    tmp_move_vector = self._apply_enhancement(
                        instance_list[i],
                        method_name, 
                        [self.geom_num_list] + base_params + [tmp_move_vector]
                    )

            move_vector_list.append(tmp_move_vector)

        return move_vector_list

    def _apply_enhancement(self, instance, method_name, params):
        """
        Helper method to apply enhancement techniques to the move vector.
        
        Parameters:
        -----------
        instance : object
            The enhancement technique instance
        method_name : str
            The name of the method to use (for logging/debugging)
        params : list
            Parameters to pass to the run method
        
        Returns:
        --------
        numpy.ndarray
            The modified move vector
        """
        # For LARS, special handling is needed as it returns a scaling factor
        if method_name == "apply_lars":
            trust_delta = instance.run(*params)
            # Last parameter is the move vector
            return params[-1] * trust_delta
        else:
            # Standard run pattern for most enhancement techniques
            return instance.run(*params)


    def calc_move_vector(self, iter, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances, projection_constrain=False, print_flag=True):#geom_num_list:Bohr
        natom = len(geom_num_list)
        
        ###
        #-------------------------------------------------------------
        geom_num_list = geom_num_list.reshape(natom*3, 1)
        B_g = B_g.reshape(natom*3, 1)
        pre_B_g = pre_B_g.reshape(natom*3, 1)
        pre_geom = pre_geom.reshape(natom*3, 1)
        g = g.reshape(natom*3, 1)
        pre_g = pre_g.reshape(natom*3, 1)
        pre_move_vector = pre_move_vector.reshape(natom*3, 1)
        initial_geom_num_list = initial_geom_num_list.reshape(natom*3, 1)
        #-------------------------------------------------------------
        ###
        self.iter = iter
        self.geom_num_list = geom_num_list

        #-------------------------------------------------------------
        # update trust radii
        #-------------------------------------------------------------
        self.update_trust_radius_conditionally(optimizer_instances, B_e, pre_B_e, pre_B_g, pre_move_vector, geom_num_list)
        self.handle_projection_constraint(projection_constrain)
        
        #---------------------------------
        #calculate move vector
        #---------------------------------

        move_vector_list = self.update_move_vector_list(optimizer_instances,
        B_g,
        pre_B_g,
        pre_geom,
        B_e,
        pre_B_e,
        pre_move_vector,
        initial_geom_num_list,
        g,
        pre_g)
        
        #---------------------------------
        # switch step update method
        #---------------------------------
        move_vector, optimizer_instances = self.switch_move_vector(B_g,
        move_vector_list,
        optimizer_instances,
        max_rms_force_switching_threshold=0.05,
        min_rms_force_switching_threshold=0.005
        )

        if print_flag:
            print("==================================================================================")
 
        if np.linalg.norm(move_vector) > self.trust_radii:
            move_vector = self.trust_radii * move_vector/np.linalg.norm(move_vector)
        
        if print_flag:
            print("trust radii (unit. ang.): ", self.trust_radii)
            print("step  radii (unit. ang.): ", np.linalg.norm(move_vector))
        new_geometry = (geom_num_list - move_vector)  
        
        new_geometry = new_geometry.reshape(natom, 3)
        move_vector = move_vector.reshape(natom, 3)
    
        for i in range(len(optimizer_instances)):
            if print_flag:
                print(f"Optimizer instance {i}: ", optimizer_instances[i])
            if self.newton_tag[i]:
                tmp_hess = optimizer_instances[i].hessian
                tmp_bias_hess = optimizer_instances[i].bias_hessian
                display_eigvals(tmp_hess, tmp_bias_hess, self.element_list, new_geometry)
        if print_flag:
            print("==================================================================================")
        new_geometry *= self.unitval.bohr2angstroms #Bohr -> ang.
        #new_lambda_list : a.u.
        #new_geometry : angstrom
        #move_vector : bohr
        #lambda_movestep : a.u.
        
        return new_geometry, np.array(move_vector, dtype="float64"), optimizer_instances



def display_eigvals(hessian, bias_hessian, element_list, geom):
   
    if bias_hessian is not None:
        tmp_hess = hessian 
        H = Calculationtools().project_out_hess_tr_and_rot_for_coord(tmp_hess + bias_hessian, element_list, geom, display_eigval=False)
    else:
        tmp_hess = hessian
        H = Calculationtools().project_out_hess_tr_and_rot_for_coord(tmp_hess, element_list, geom, display_eigval=False)
    H = (H + H.T) / 2  # Make sure H is symmetric
    evals, _ = np.linalg.eigh(H)
    
    # Filter eigenvalues with absolute value greater than 1e-10
    filtered_evals = evals[np.abs(evals) > 1e-10]
    filtered_evals = np.sort(filtered_evals)
    num_values = len(filtered_evals)
    
    print(f"EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: {num_values}):")
    for i in range(0, num_values, 6):
        line = ' '.join(f'{v:12.8f}' for v in filtered_evals[i:i+6])
        print(line)
    
    