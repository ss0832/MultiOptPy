import numpy as np
import copy

from parameter import UnitValueLib, atomic_mass
from calc_tools import Calculationtools

from Optimizer.adabelief import Adabelief
#from Optimizer.fastadabelief import FastAdabelief
#from Optimizer.adaderivative import Adaderivative
#from Optimizer.sadam import SAdam
#from Optimizer.samsgrad import SAMSGrad
#from Optimizer.QHAdam import QHAdam
#from Optimizer.adamax import AdaMax
#from Optimizer.yogi import YOGI
#from Optimizer.nadam import NAdam
from Optimizer.fire import FIRE
from Optimizer.abc_fire import ABC_FIRE
from Optimizer.fire2 import FIRE2
#from Optimizer.adadiff import AdaDiff
#from Optimizer.adamod import Adamod
from Optimizer.radam import RADAM
from Optimizer.eve import EVE
#from Optimizer.adamw import AdamW
from Optimizer.adam import Adam
#from Optimizer.adafactor import Adafactor
from Optimizer.prodigy import Prodigy
#from Optimizer.adabound import AdaBound
#from Optimizer.adadelta import Adadelta
from Optimizer.conjugate_gradient import ConjgateGradient
#from Optimizer.hybrid_rfo import HybridRFO
from Optimizer.rfo import RationalFunctionOptimization
#from Optimizer.ric_rfo import RedundantInternalRFO
from Optimizer.rsprfo import RSPRFO, EnhancedRSPRFO
from Optimizer.rsirfo import RSIRFO
#from Optimizer.newton import Newton
from Optimizer.lbfgs import LBFGS
from Optimizer.tr_lbfgs import TRLBFGS
#from Optimizer.rmspropgrave import RMSpropGrave
from Optimizer.lookahead import LookAhead
from Optimizer.lars import LARS
from Optimizer.gdiis import GDIIS
from Optimizer.ediis import EDIIS
from Optimizer.gediis import GEDIIS
from Optimizer.c2diis import C2DIIS
from Optimizer.adiis import ADIIS 
from Optimizer.kdiis import KrylovDIIS as KDIIS
from Optimizer.gpr_step import GPRStep
from Optimizer.gan_step import GANStep
from Optimizer.rl_step import RLStepSizeOptimizer
from Optimizer.component_wise_scaling import ComponentWiseScaling
from Optimizer.coordinate_locking import CoordinateLocking
from Optimizer.trust_radius import TrustRadius
from Optimizer.gradientdescent import GradientDescent, MassWeightedGradientDescent
from Optimizer.gpmin import GPmin
from Optimizer.cubic_newton import CubicNewton

optimizer_mapping = {
    "adabelief": Adabelief,
    #"fastadabelief": FastAdabelief,
    "radam": RADAM,
    #"adamod": Adamod,
    #"yogi": YOGI,
    #"sadam": SAdam,
    #"qhadam": QHAdam,
    #"samsgrad": SAMSGrad,
    #"adadelta": Adadelta,
    #"adamw": AdamW,
    #"adadiff": AdaDiff,
    #"adafactor": Adafactor,
    "eve": EVE,
    "prodigy": Prodigy,
    #"adamax": AdaMax,
    #"nadam": NAdam,
    #"rmspropgrave": RMSpropGrave,
    "abcfire": ABC_FIRE,
    "fire2": FIRE2,
    "fire": FIRE,
    #"adaderivative": Adaderivative,
    "mwgradientdescent": MassWeightedGradientDescent,
    "gradientdescent": GradientDescent,
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
    "cubicnewton_bfgs": {"delta": 0.50},
    "cubicnewton_fsb": {"delta": 0.50},
    "cubicnewton_bofill": {"delta": 0.50},
    "cubicnewton_msp": {"delta": 0.50},
    "cubicnewton_sr1": {"delta": 0.50},
    "cubicnewton_psb": {"delta": 0.50},
    "cubicnewton_flowchart": {"delta": 0.50},


    "rsirfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_msp": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_psb": {"delta": 0.50, "rfo_type": 1},
    "rsirfo_flowchart": {"delta": 0.50, "rfo_type": 1},
    

    "ersprfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "ersprfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "ersprfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "ersprfo_msp": {"delta": 0.50, "rfo_type": 1},
    "ersprfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "ersprfo_psb": {"delta": 0.50, "rfo_type": 1},
    "ersprfo_flowchart": {"delta": 0.50, "rfo_type": 1},
    
    "rsprfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_msp": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_psb": {"delta": 0.50, "rfo_type": 1},
    "rsprfo_flowchart": {"delta": 0.50, "rfo_type": 1},

    "sirfo3_bfgs": {"delta": 0.50, "rfo_type": 3},
    "sirfo3_fsb": {"delta": 0.50, "rfo_type": 3},
    "sirfo3_bofill": {"delta": 0.50, "rfo_type": 3},
    "sirfo3_msp": {"delta": 0.50, "rfo_type": 3},
    "sirfo3_sr1": {"delta": 0.50, "rfo_type": 3},
    "sirfo3_psb": {"delta": 0.50, "rfo_type": 3},
    "sirfo3_flowchart": {"delta": 0.50, "rfo_type": 3},
    
    "rfo3_bfgs": {"delta": 0.50, "rfo_type": 3},
    "rfo3_fsb": {"delta": 0.50, "rfo_type": 3},
    "rfo3_bofill": {"delta": 0.50, "rfo_type": 3},
    "rfo3_msp": {"delta": 0.50, "rfo_type": 3},
    "rfo3_sr1": {"delta": 0.50, "rfo_type": 3},
    "rfo3_psb": {"delta": 0.50, "rfo_type": 3},
    "rfo3_flowchart": {"delta": 0.50, "rfo_type": 3},
    
    "rfo2_bfgs": {"delta": 0.50, "rfo_type": 2},
    "rfo2_fsb": {"delta": 0.50, "rfo_type": 2},
    "rfo2_bofill": {"delta": 0.50, "rfo_type": 2},
    "rfo2_msp": {"delta": 0.50, "rfo_type": 2},
    "rfo2_sr1": {"delta": 0.50, "rfo_type": 2},
    "rfo2_psb": {"delta": 0.50, "rfo_type": 2},
    "rfo2_flowchart": {"delta": 0.50, "rfo_type": 2},
    
    
    
    "mrfo_bfgs": {"delta": 0.30, "rfo_type": 1},
    "mrfo_fsb": {"delta": 0.30, "rfo_type": 1},
    "mrfo_bofill": {"delta": 0.30, "rfo_type": 1},
    "mrfo_msp": {"delta": 0.30, "rfo_type": 1},
    "mrfo_sr1": {"delta": 0.30, "rfo_type": 1},
    "mrfo_psb": {"delta": 0.30, "rfo_type": 1},
    "mrfo_flowchart": {"delta": 0.30, "rfo_type": 1},

 
    "rfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "rfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "rfo_msp": {"delta": 0.50, "rfo_type": 1},
    "rfo_sr1": {"delta": 0.50, "rfo_type": 1},
    "rfo_psb": {"delta": 0.50, "rfo_type": 1},
    "rfo_flowchart": {"delta": 0.50, "rfo_type": 1},
    
}



class CalculateMoveVector:
    def __init__(self, DELTA, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0, model_hess_flag=None, max_trust_radius=None):
        self.DELTA = DELTA
        self.temperature = temperature
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.unitval = UnitValueLib()
        self.FC_COUNT = FC_COUNT
        self.MAX_MAX_FORCE_SWITCHING_THRESHOLD = 0.0050
        self.MIN_MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.MAX_RMS_FORCE_SWITCHING_THRESHOLD = 0.05
        self.MIN_RMS_FORCE_SWITCHING_THRESHOLD = 0.005
        self.max_trust_radius = max_trust_radius
        self.CALC_TRUST_RADII = TrustRadius()
        if self.max_trust_radius is not None:
            if self.max_trust_radius <= 0.0:
                print("max_trust_radius must be greater than 0.0")
                exit()
                
            self.CALC_TRUST_RADII.set_max_trust_radius(self.max_trust_radius)
        
        self.trust_radii = 0.5
        self.saddle_order = saddle_order
        self.iter = 0
        self.element_list = element_list
        self.model_hess_flag = model_hess_flag

        
    def initialization(self, method):
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
        

        for i, m in enumerate(method):
            lower_m = m.lower()

            if lower_m in specific_cases:
                case = specific_cases[lower_m]
                optimizer_instances.append(case["optimizer"]())
                newton_tag.append(False)
                lookahead_instances.append(case["lookahead"])
                lars_instances.append(case["lars"])
                gdiis_instances.append(GDIIS() if "gdiis" in lower_m else None)
                if "gediis" in lower_m:
                    gediis_instances.append(GEDIIS())
                    ediis_instances.append(None)
                else:
                    ediis_instances.append(EDIIS() if "ediis" in lower_m else None)
                    gediis_instances.append(None)
                c2diis_instances.append(C2DIIS() if "c2diis" in lower_m else None)
                adiis_instances.append(ADIIS() if "adiis" in lower_m else None)
                kdiis_instances.append(KDIIS() if "kdiis" in lower_m else None)
                
                coordinate_locking_instances.append(CoordinateLocking() if "coordinate_locking" in lower_m else None)
                coordinate_wise_scaling_instances.append(ComponentWiseScaling() if "component_wise_scaling" in lower_m else None)
                
                gpr_step_instances.append(GPRStep() if "gpr_step" in lower_m else None)
                gan_step_instances.append(GANStep() if "gan_step" in lower_m else None)
                rl_step_instances.append(RLStepSizeOptimizer() if "rl_step" in lower_m else None)
                
                       
                
            elif any(key in lower_m for key in optimizer_mapping):
                for key, optimizer_class in optimizer_mapping.items():
                    if key in lower_m:
                        optimizer_instances.append(optimizer_class())
                        if lower_m == "mwgradientdescent":#Eulur method to calculate IRC path.
                            optimizer_instances[i].element_list = self.element_list
                            optimizer_instances[i].atomic_mass = atomic_mass # function in parameter.py
                        
                        
                        newton_tag.append(False)
                        lookahead_instances.append(LookAhead() if "lookahead" in lower_m else None)
                        lars_instances.append(LARS() if "lars" in lower_m else None)
                        gdiis_instances.append(GDIIS() if "gdiis" in lower_m else None)
                        if "gediis" in lower_m:
                            gediis_instances.append(GEDIIS())
                            ediis_instances.append(None)
                        else:
                            ediis_instances.append(EDIIS() if "ediis" in lower_m else None)
                            gediis_instances.append(None)
                        c2diis_instances.append(C2DIIS() if "c2diis" in lower_m else None)
                        adiis_instances.append(ADIIS() if "adiis" in lower_m else None)
                        kdiis_instances.append(KDIIS() if "kdiis" in lower_m else None)
                     
                        coordinate_locking_instances.append(CoordinateLocking() if "coordinate_locking" in lower_m else None)
                        coordinate_wise_scaling_instances.append(ComponentWiseScaling() if "component_wise_scaling" in lower_m else None)
                        
                        gpr_step_instances.append(GPRStep() if "gpr_step" in lower_m else None)
                        gan_step_instances.append(GANStep() if "gan_step" in lower_m else None)
                        rl_step_instances.append(RLStepSizeOptimizer() if "rl_step" in lower_m else None)
                       
                        break
                    
            elif lower_m in ["cg", "cg_pr", "cg_fr", "cg_hs", "cg_dy"]:
                optimizer_instances.append(ConjgateGradient(method=m))
                newton_tag.append(False)
                lookahead_instances.append(None)
                lars_instances.append(None)
                gdiis_instances.append(GDIIS() if "gdiis" in lower_m else None)
                kdiis_instances.append(KDIIS() if "kdiis" in lower_m else None)
                if "gediis" in lower_m:
                    gediis_instances.append(GEDIIS())
                    ediis_instances.append(None)
                else:
                    ediis_instances.append(EDIIS() if "ediis" in lower_m else None)
                    gediis_instances.append(None)
                c2diis_instances.append(C2DIIS() if "c2diis" in lower_m else None)
                adiis_instances.append(ADIIS() if "adiis" in lower_m else None)
                
                coordinate_locking_instances.append(CoordinateLocking() if "coordinate_locking" in lower_m else None)
                coordinate_wise_scaling_instances.append(ComponentWiseScaling() if "component_wise_scaling" in lower_m else None)
                
                gpr_step_instances.append(GPRStep() if "gpr_step" in lower_m else None)
                gan_step_instances.append(GANStep() if "gan_step" in lower_m else None)
                rl_step_instances.append(RLStepSizeOptimizer() if "rl_step" in lower_m else None)
           
            
            elif any(key in lower_m for key in quasi_newton_mapping):
                for key, settings in quasi_newton_mapping.items():
                    if key in lower_m:
                        print(key)
                        if "ersprfo" in key:
                            optimizer_instances.append(EnhancedRSPRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list))
                        elif "cubicnewton" in key:
                            optimizer_instances.append(CubicNewton(method=m, saddle_order=self.saddle_order, element_list=self.element_list))
                        
                        elif "rsprfo" in key:
                            optimizer_instances.append(RSPRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list))

                        elif "rsirfo" in key:
                            optimizer_instances.append(RSIRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list))   
                        elif "rfo" in key:
                            optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order, trust_radius=self.trust_radii, element_list=self.element_list))
                        else:
                            print("This method is not implemented. :", m, " Thus, exiting.")
                            exit()
                            

                        optimizer_instances[i].DELTA = settings["delta"]
                        if "linesearch" in settings:
                            optimizer_instances[i].linesearchflag = True
                        newton_tag.append(True)
                        lookahead_instances.append(LookAhead() if "lookahead" in lower_m else None)
                        lars_instances.append(LARS() if "lars" in lower_m else None)
                        gdiis_instances.append(GDIIS() if "gdiis" in lower_m else None)
                        kdiis_instances.append(KDIIS() if "kdiis" in lower_m else None)
                        if "gediis" in lower_m:
                            gediis_instances.append(GEDIIS())
                            ediis_instances.append(None)
                        else:
                            ediis_instances.append(EDIIS() if "ediis" in lower_m else None)
                            gediis_instances.append(None)
                        c2diis_instances.append(C2DIIS() if "c2diis" in lower_m else None)
                        adiis_instances.append(ADIIS() if "adiis" in lower_m else None)
                       
                        coordinate_locking_instances.append(CoordinateLocking() if "coordinate_locking" in lower_m else None)
                        coordinate_wise_scaling_instances.append(ComponentWiseScaling() if "component_wise_scaling" in lower_m else None)
                        
                        gpr_step_instances.append(GPRStep() if "gpr_step" in lower_m else None)
                        gan_step_instances.append(GANStep() if "gan_step" in lower_m else None)
                        rl_step_instances.append(RLStepSizeOptimizer() if "rl_step" in lower_m else None)
                       
                        break
            else:
                print("This method is not implemented. :", m, " Thus, Default method is used.")
                optimizer_instances.append(FIRE())
                newton_tag.append(False)
                lookahead_instances.append(None)
                lars_instances.append(None)
                gdiis_instances.append(None)
                ediis_instances.append(None)
                gediis_instances.append(None)
                c2diis_instances.append(None)
                adiis_instances.append(None)
                kdiis_instances.append(None)
                coordinate_locking_instances.append(None)
                coordinate_wise_scaling_instances.append(None)
                gpr_step_instances.append(None)
                gan_step_instances.append(None)
                rl_step_instances.append(None)
            
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
        if not (model_hess is None and self.FC_COUNT == -1):
            self.trust_radii = self.CALC_TRUST_RADII.update_trust_radii(
                B_e, pre_B_e, pre_B_g, pre_move_vector, model_hess, geom_num_list, self.trust_radii
            )

        if self.max_trust_radius is not None:
            print("user_difined_max_trust_radius: ", self.max_trust_radius)
        else:
            # If saddle order is positive, constrain the trust radii
            if self.saddle_order > 0:
                self.trust_radii = min(self.trust_radii, 0.1)

            # If this is the first iteration but not full-coordinate -1 check
            if self.iter == 0 and self.FC_COUNT != -1:
                if self.saddle_order > 0:
                    self.trust_radii = min(self.trust_radii, 0.1)

    def handle_projection_constraint(self, projection_constrain):
        """
        Constrain the trust radii if projection constraint is enabled.
        """
        if projection_constrain:
            if self.max_trust_radius is not None:
                pass
            else:
                self.trust_radii = min(self.trust_radii, 0.1)



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
            pre_g):
        """
        Update a list of move vectors from multiple optimizer instances,
        including optional enhancements through various techniques.
        """
        move_vector_list = []

        # Enhancement techniques and their parameter configurations
        # Format: [list_of_instances, method_name, [parameters]]
        enhancement_config = [
            # Base optimizer - separate handling
            
            # Trust region and momentum techniques
            [self.lars_instances, "apply_lars", [B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, 
                                                initial_geom_num_list, g, pre_g]],
            [self.lookahead_instances, "apply_lookahead", [B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector,
                                                        initial_geom_num_list, g, pre_g]],
            
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
                _ = Calculationtools().project_out_hess_tr_and_rot_for_coord( #hessian, element_list, geometry
                    optimizer_instances[i].hessian + optimizer_instances[i].bias_hessian,
                    self.element_list,
                    new_geometry)
        if print_flag:
            print("==================================================================================")
        new_geometry *= self.unitval.bohr2angstroms #Bohr -> ang.
        #new_lambda_list : a.u.
        #new_geometry : angstrom
        #move_vector : bohr
        #lambda_movestep : a.u.
        
        return new_geometry, np.array(move_vector, dtype="float64"), optimizer_instances

