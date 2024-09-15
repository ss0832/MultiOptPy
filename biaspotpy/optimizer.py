import numpy as np
import copy

from parameter import UnitValueLib

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
from Optimizer.adabound import AdaBound
from Optimizer.adadelta import Adadelta
from Optimizer.purtubation import Perturbation
from Optimizer.conjugate_gradient import ConjgateGradient
from Optimizer.rfo import RationalFunctionOptimization 
from Optimizer.newton import Newton 
from Optimizer.rmspropgrave import RMSpropGrave
from Optimizer.lookahead import LookAhead
from Optimizer.lars import LARS
from Optimizer.trust_radius import update_trust_radii

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
    "adabound": AdaBound,
    "eve": EVE,
    "prodigy": Prodigy,
    "adamax": AdaMax,
    "nadam": NAdam,
    "rmspropgrave": RMSpropGrave,
    "fire": FIRE,
    "adaderivative": Adaderivative,
}

specific_cases = {
    "ranger": {"optimizer": RADAM, "lookahead": LookAhead(), "lars": None},
    "rangerlars": {"optimizer": RADAM, "lookahead": LookAhead(), "lars": LARS()},
    "adam": {"optimizer": Adam, "lookahead": LookAhead(), "lars": LARS()},
    "adamlars": {"optimizer": Adam, "lookahead": None, "lars": LARS()},
    "adamlookahead": {"optimizer": Adam, "lookahead": LookAhead(), "lars": None},
    "adamlookaheadlars": {"optimizer": Adam, "lookahead": LookAhead(), "lars": LARS()},
}

quasi_newton_mapping = {
    "rfo3_bfgs": {"delta": 0.50, "rfo_type": 3},
    "rfo3_fsb": {"delta": 0.50, "rfo_type": 3},
    "rfo3_bofill": {"delta": 0.50, "rfo_type": 3},
    "rfo3_msp": {"delta": 0.50, "rfo_type": 3},
    "rfo2_bfgs": {"delta": 0.50, "rfo_type": 2},
    "rfo2_fsb": {"delta": 0.50, "rfo_type": 2},
    "rfo2_bofill": {"delta": 0.50, "rfo_type": 2},
    "rfo2_msp": {"delta": 0.50, "rfo_type": 2},
    "mrfo_bfgs": {"delta": 0.30, "rfo_type": 1},
    "mrfo_fsb": {"delta": 0.30, "rfo_type": 1},
    "mrfo_bofill": {"delta": 0.30, "rfo_type": 1},
    "mrfo_msp": {"delta": 0.30, "rfo_type": 1},
    "rfo_bfgs": {"delta": 0.50, "rfo_type": 1},
    "rfo_fsb": {"delta": 0.50, "rfo_type": 1},
    "rfo_bofill": {"delta": 0.50, "rfo_type": 1},
    "rfo_msp": {"delta": 0.50, "rfo_type": 1},
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
    def __init__(self, DELTA, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0):
        self.DELTA = DELTA
        self.temperature = temperature
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.unitval = UnitValueLib()
        self.FC_COUNT = FC_COUNT
        self.MAX_MAX_FORCE_SWITCHING_THRESHOLD = 0.0050
        self.MIN_MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.MAX_RMS_FORCE_SWITCHING_THRESHOLD = 0.05
        self.MIN_RMS_FORCE_SWITCHING_THRESHOLD = 0.005
        self.trust_radii = 1.0
        self.saddle_order = saddle_order
        self.iter = 0
        self.element_list = element_list
        self.trust_radii_update = "trust"
        
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
                        if "rfo" in key:
                            optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order))
                        else:
                            optimizer_instances.append(Newton(method=m))
                        optimizer_instances[i].DELTA = settings["delta"]
                        if "linesearch" in settings:
                            optimizer_instances[i].linesearchflag = settings["linesearch"]
                        newton_tag.append(True)
                        lookahead_instances.append(None)
                        lars_instances.append(None)
                        break
            else:
                print("This method is not implemented. :", m, " Thus, Default method is used.")
                optimizer_instances.append(Adabelief())
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

  
    def calc_move_vector(self, iter, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances):#geom_num_list:Bohr
        self.iter = iter
        self.geom_num_list = geom_num_list
        move_vector_list = []

        #-------------------------------------------------------------
        # update trust radii
        #-------------------------------------------------------------
        if self.iter == 0 and self.FC_COUNT != -1:
            if self.saddle_order > 0:
                self.trust_radii = min(self.trust_radii, 0.1)
  
        elif self.FC_COUNT == -1:
            pass

        else:
            if self.trust_radii_update == "trust":
                for i in range(len(optimizer_instances)):
                    if self.newton_tag[i]:
                        model_hess = optimizer_instances[i].hessian + optimizer_instances[i].bias_hessian
                        break
                else:
                    model_hess = None
            if model_hess is None:
                pass
            else:
                self.trust_radii = update_trust_radii(B_e, pre_B_e, pre_B_g, pre_move_vector, model_hess, geom_num_list, self.trust_radii, self.trust_radii_update)
            
            if self.saddle_order > 0:
                self.trust_radii = min(self.trust_radii, 0.1)

        #if self.iter == 20:
        #   raise ValueError("stop")
        #---------------------------------
        #calculate move vector
        #---------------------------------
        for i in range(len(optimizer_instances)):
            tmp_move_vector = optimizer_instances[i].run(self.geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            tmp_move_vector = np.array(tmp_move_vector, dtype="float64")
            if self.lars_instances[i] is not None:
                trust_delta = self.lars_instances[i].run(self.geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, tmp_move_vector)
                
                tmp_move_vector = tmp_move_vector * trust_delta
                
            if self.lookahead_instances[i] is not None:
                tmp_move_vector = self.lookahead_instances[i].run(self.geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, tmp_move_vector)
            
            move_vector_list.append(tmp_move_vector)
            
        
        #---------------------------------
        # switch step update method
        #---------------------------------
        if len(move_vector_list) > 1:
            if abs(np.sqrt(np.square(B_g).mean())) > self.MAX_RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method:", self.method[0])

            elif abs(np.sqrt(np.square(B_g).mean())) <= self.MAX_RMS_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) > self.MIN_RMS_FORCE_SWITCHING_THRESHOLD: 
                x_i = abs(np.sqrt(np.square(B_g).mean()))
                x_max = self.MAX_RMS_FORCE_SWITCHING_THRESHOLD
                x_min = self.MIN_RMS_FORCE_SWITCHING_THRESHOLD

                x_j = (x_i - x_min) / (x_max - x_min)
                
                f_val = 1 / (1 + np.exp(-10.0 * (x_j - 0.5)))
                move_vector = np.array(move_vector_list[0], dtype="float64") * f_val + np.array(move_vector_list[1], dtype="float64") * (1.0 - f_val)
                print(f_val, x_j)

            
            else:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method:", self.method[1])

        else:
            move_vector = copy.copy(move_vector_list[0])
            if self.newton_tag[0]:
                self.diag_hess_and_display(optimizer_instances[0])
                
        # add perturbation (toy function)
        P = Perturbation(temperature=self.temperature)
        perturbation = P.boltzmann_dist_perturb(move_vector)
        
        move_vector += perturbation
        print("perturbation: ", np.linalg.norm(perturbation))

        #-------------------------------------------------------------
        #display trust radii
        #-------------------------------------------------------------
        if np.linalg.norm(move_vector) > self.trust_radii:
            move_vector = self.trust_radii * move_vector/np.linalg.norm(move_vector)
        print("trust radii: ", self.trust_radii)
        print("step  radii: ", np.linalg.norm(move_vector))

        new_geometry = (geom_num_list - move_vector) * self.unitval.bohr2angstroms 
        #---------------------------------
        print("Optimizer instances: ", optimizer_instances)
        return new_geometry, np.array(move_vector, dtype="float64"), optimizer_instances


