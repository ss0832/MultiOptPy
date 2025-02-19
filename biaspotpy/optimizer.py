import numpy as np
import copy

from parameter import UnitValueLib, atomic_mass

from Optimizer.adabelief import Adabelief
from Optimizer.fastadabelief import FastAdabelief
from Optimizer.adaderivative import Adaderivative
from Optimizer.sadam import SAdam
from Optimizer.samsgrad import SAMSGrad
from Optimizer.QHAdam import QHAdam
from Optimizer.adamax import AdaMax
from Optimizer.yogi import YOGI
from Optimizer.nadam import NAdam
from Optimizer.fire import FIRE
from Optimizer.adadiff import AdaDiff
from Optimizer.adamod import Adamod
from Optimizer.radam import RADAM
from Optimizer.eve import EVE
from Optimizer.adamw import AdamW
from Optimizer.adam import Adam
from Optimizer.adafactor import Adafactor
from Optimizer.prodigy import Prodigy
#from Optimizer.adabound import AdaBound
from Optimizer.adadelta import Adadelta
from Optimizer.conjugate_gradient import ConjgateGradient
from Optimizer.hybrid_rfo import HybridCoordinateAugmentedRFO 
from Optimizer.rfo import RationalFunctionOptimization 
from Optimizer.newton import Newton 
from Optimizer.rmspropgrave import RMSpropGrave
from Optimizer.lookahead import LookAhead
from Optimizer.lars import LARS
from Optimizer.trust_radius import update_trust_radii
from Optimizer.gradientdescent import GradientDescent, MassWeightedGradientDescent

optimizer_mapping = {
    "adabelief": Adabelief,
    "fastadabelief": FastAdabelief,
    "radam": RADAM,
    "adamod": Adamod,
    "yogi": YOGI,
    "sadam": SAdam,
    "qhadam": QHAdam,
    "samsgrad": SAMSGrad,
    "adadelta": Adadelta,
    "adamw": AdamW,
    "adadiff": AdaDiff,
    "adafactor": Adafactor,
    "eve": EVE,
    "prodigy": Prodigy,
    "adamax": AdaMax,
    "nadam": NAdam,
    "rmspropgrave": RMSpropGrave,
    "fire": FIRE,
    "adaderivative": Adaderivative,
    "mwgradientdescent": MassWeightedGradientDescent,
    "gradientdescent": GradientDescent,
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
    "rfo4_bfgs": {"delta": 0.50, "rfo_type": 4},
    "rfo4_fsb": {"delta": 0.50, "rfo_type": 4},
    "rfo4_bofill": {"delta": 0.50, "rfo_type": 4},
    "rfo4_msp": {"delta": 0.50, "rfo_type": 4},
    "rfo4_sr1": {"delta": 0.50, "rfo_type": 4},
    "rfo4_psb": {"delta": 0.50, "rfo_type": 4},
    "rfo4_flowchart": {"delta": 0.50, "rfo_type": 4},
    
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
    
    "hybrid_rfo_fsb": {"delta": 0.05, "rfo_type": 1},
    "hybrid_rfo_bofill": {"delta": 0.05, "rfo_type": 1},
    "hybrid_rfo_msp": {"delta": 0.05, "rfo_type": 1},
    "hybrid_rfo_bfgs": {"delta": 0.05, "rfo_type": 1},
    
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
    
    "bfgs": {"delta": 0.10, "linesearch": False},
    "fsb": {"delta": 0.10, "linesearch": False},
    "bofill": {"delta": 0.10, "linesearch": False},
    "msp": {"delta": 0.10, "linesearch": False},
    "bfgs_ls": {"delta": 0.50, "linesearch": True},
    "fsb_ls": {"delta": 0.50, "linesearch": True},
    "bofill_ls": {"delta": 0.50, "linesearch": True},
    "msp_ls": {"delta": 0.50, "linesearch": True},

}



class CalculateMoveVector:
    def __init__(self, DELTA, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0, model_hess_flag=False):
        self.DELTA = DELTA
        self.temperature = temperature
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.unitval = UnitValueLib()
        self.FC_COUNT = FC_COUNT
        self.MAX_MAX_FORCE_SWITCHING_THRESHOLD = 0.0050
        self.MIN_MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.MAX_RMS_FORCE_SWITCHING_THRESHOLD = 0.05
        self.MIN_RMS_FORCE_SWITCHING_THRESHOLD = 0.005
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

        for i, m in enumerate(method):
            lower_m = m.lower()

            if lower_m in specific_cases:
                case = specific_cases[lower_m]
                optimizer_instances.append(case["optimizer"]())
                newton_tag.append(False)
                lookahead_instances.append(case["lookahead"])
                lars_instances.append(case["lars"])
            elif any(key in lower_m for key in optimizer_mapping):
                for key, optimizer_class in optimizer_mapping.items():
                    if key in lower_m:
                        optimizer_instances.append(optimizer_class())
                        if lower_m == "mwgradientdescent":
                            optimizer_instances[i].element_list = self.element_list
                            optimizer_instances[i].atomic_mass = atomic_mass # function in parameter.py
                        
                        
                        newton_tag.append(False)
                        lookahead_instances.append(LookAhead() if "lookahead" in lower_m else None)
                        lars_instances.append(LARS() if "lars" in lower_m else None)
                        break
                    
            elif m in ["CG", "CG_PR", "CG_FR", "CG_HS", "CG_DY"]:
                optimizer_instances.append(ConjgateGradient(method=m))
                newton_tag.append(False)
                lookahead_instances.append(None)
                lars_instances.append(None)
                
            elif any(key in lower_m for key in quasi_newton_mapping):
                for key, settings in quasi_newton_mapping.items():
                    if key in lower_m:
                        print(key)
                        if "hybrid_rfo" in key:
                            optimizer_instances.append(HybridCoordinateAugmentedRFO(method=m, saddle_order=self.saddle_order, element_list=self.element_list))          
                        elif "rfo" in key:
                            optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order))
                        else:
                            optimizer_instances.append(Newton(method=m))
                        optimizer_instances[i].DELTA = settings["delta"]
                        if "linesearch" in settings:
                            optimizer_instances[i].linesearchflag = True
                        newton_tag.append(True)
                        lookahead_instances.append(None)
                        lars_instances.append(None)
                        break
            else:
                print("This method is not implemented. :", m, " Thus, Default method is used.")
                optimizer_instances.append(FIRE())
                newton_tag.append(False)
                lookahead_instances.append(None)
                lars_instances.append(None)

        self.method = method
        self.newton_tag = newton_tag
        self.lookahead_instances = lookahead_instances
        self.lars_instances = lars_instances
        #print("Optimizer instances: ", optimizer_instances)
        #print("Newton tag: ", newton_tag)
        #print("Lookahead instances: ", lookahead_instances)
        #print("LARS instances: ", lars_instances)
        #print("Method: ", method)
        
        return optimizer_instances
            

    def diag_hess_and_display(self, optimizer_instance):
        #------------------------------------------------------------
        # diagonilize hessian matrix and display eigenvalues
        #-----------------------------------------------------------
        hess_eigenvalue, _ = np.linalg.eig(optimizer_instance.hessian + optimizer_instance.bias_hessian)
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)#not display imagnary values 
        print("NORMAL MODE EIGENVALUE:\n",np.sort(hess_eigenvalue),"\n")
        return

    def update_trust_radius_conditionally(self, optimizer_instances, B_e, pre_B_e, pre_B_g, pre_move_vector, geom_num_list):
        """
        Refactored method to handle trust radius updates and conditions.
        """
        # Early exit if no full-coordinate count and no Hessian flag
        if self.FC_COUNT == -1 and not self.model_hess_flag:
            return

        # Determine if there's a model Hessian to use
        model_hess = None
        for i in range(len(optimizer_instances)):
            if self.newton_tag[i]:
                model_hess = optimizer_instances[i].hessian + optimizer_instances[i].bias_hessian
                break

        # Update trust radii only if we have a Hessian or if FC_COUNT is not -1
        if not (model_hess is None and self.FC_COUNT == -1):
            self.trust_radii = update_trust_radii(
                B_e, pre_B_e, pre_B_g, pre_move_vector, model_hess, geom_num_list, self.trust_radii
            )

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
            self.trust_radii = min(self.trust_radii, 0.1)



    def switch_move_vector(self,
        B_g,
        move_vector_list,
        method_list=None,
        max_rms_force_switching_threshold=0.5,
        min_rms_force_switching_threshold=0.1,
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
        Refactored method to update a list of move vectors from multiple optimizer instances,
        including optional LARS and LookAhead adjustments.
        """
        move_vector_list = []

        for i, optimizer_instance in enumerate(optimizer_instances):
            # Obtain initial move vector
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

            # Apply LARS adjustment if available
            if self.lars_instances[i] is not None:
                trust_delta = self.lars_instances[i].run(
                    self.geom_num_list,
                    B_g,
                    pre_B_g,
                    pre_geom,
                    B_e,
                    pre_B_e,
                    pre_move_vector,
                    initial_geom_num_list,
                    g,
                    pre_g,
                    tmp_move_vector
                )
                tmp_move_vector *= trust_delta

            # Apply LookAhead adjustment if available
            if self.lookahead_instances[i] is not None:
                tmp_move_vector = self.lookahead_instances[i].run(
                    self.geom_num_list,
                    B_g,
                    pre_B_g,
                    pre_geom,
                    B_e,
                    pre_B_e,
                    pre_move_vector,
                    initial_geom_num_list,
                    g,
                    pre_g,
                    tmp_move_vector
                )

            move_vector_list.append(tmp_move_vector)

        return move_vector_list


    def calc_move_vector(self, iter, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances, projection_constrain=False):#geom_num_list:Bohr
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
        max_rms_force_switching_threshold=0.5,
        min_rms_force_switching_threshold=0.1
        )
        
        #-------------------------------------------------------------
        #display trust radii
        #-------------------------------------------------------------
        if np.linalg.norm(move_vector) > self.trust_radii:
            move_vector = self.trust_radii * move_vector/np.linalg.norm(move_vector)
        print("trust radii: ", self.trust_radii)
        print("step  radii: ", np.linalg.norm(move_vector))

        new_geometry = (geom_num_list - move_vector) * self.unitval.bohr2angstroms #Bohr -> ang.
        #---------------------------------
        print("Optimizer instances: ", optimizer_instances)
        ###
        #-------------------------------------------------------------
   
        new_geometry = new_geometry.reshape(natom, 3)
        move_vector = move_vector.reshape(natom, 3)
        #-------------------------------------------------------------
      
        
        #new_lambda_list : a.u.
        #new_geometry : angstrom
        #move_vector : bohr
        #lambda_movestep : a.u.
        return new_geometry, np.array(move_vector, dtype="float64"), optimizer_instances

