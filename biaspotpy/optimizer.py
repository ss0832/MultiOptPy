import numpy as np
import copy

from parameter import UnitValueLib
from calc_tools import Calculationtools


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


class CalculateMoveVector:
    def __init__(self, DELTA, trust_radii, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0):
        self.DELTA = DELTA
        self.temperature = temperature
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.unitval = UnitValueLib()
        self.FC_COUNT = FC_COUNT
        self.MAX_MAX_FORCE_SWITCHING_THRESHOLD = 0.0050
        self.MIN_MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.MAX_RMS_FORCE_SWITCHING_THRESHOLD = 0.05
        self.MIN_RMS_FORCE_SWITCHING_THRESHOLD = 0.005
        self.trust_radii = trust_radii
        self.saddle_order = saddle_order
        self.iter = 0
        self.element_list = element_list
        self.trust_radii_update = "legacy"
    
    def initialization(self, method):
        optimizer_instances = []
        newton_tag = []
        for i, m in enumerate(method):
            # group of steepest descent
            if m == "AdaBelief":
                optimizer_instances.append(Adabelief())
                newton_tag.append(False)
            elif m == "FastAdaBelief":
                optimizer_instances.append(FastAdabelief())
                newton_tag.append(False)
            elif m == "RADAM":
                optimizer_instances.append(RADAM())
                newton_tag.append(False)
            elif m == "Adamod":
                optimizer_instances.append(Adamod())       
                newton_tag.append(False)                   
            elif m == "YOGI":
                optimizer_instances.append(YOGI())      
                newton_tag.append(False)            
            elif m == "SADAM":
                optimizer_instances.append(SAdam()) 
                newton_tag.append(False)
            elif m == "QHADAM":
                optimizer_instances.append(QHAdam())
                newton_tag.append(False)
            elif m == "SAMSGrad":
                optimizer_instances.append(SAMSGrad())
                newton_tag.append(False)
            elif m == "Adam":
                optimizer_instances.append(Adam()) 
                newton_tag.append(False)
            elif m == "Adadelta":
                optimizer_instances.append(Adadelta())
                newton_tag.append(False)
            elif m == "AdamW":
                optimizer_instances.append(AdamW())
                newton_tag.append(False)
            elif m == "AdaDiff":
                optimizer_instances.append(AdaDiff())
                newton_tag.append(False)
            elif m == "Adafactor":
                optimizer_instances.append(Adafactor())
                newton_tag.append(False)
            elif m == "Adabound":
                optimizer_instances.append(AdaBound())
                newton_tag.append(False)
            elif m == "EVE":
                optimizer_instances.append(EVE())
                newton_tag.append(False)
            elif m == "Prodigy":
                optimizer_instances.append(Prodigy())
                newton_tag.append(False)
            elif m == "AdaMax":
                optimizer_instances.append(AdaMax())
                newton_tag.append(False)
            elif m == "NAdam":    
                optimizer_instances.append(NAdam())
                newton_tag.append(False)
            elif m == "RMSpropGrave":    
                optimizer_instances.append(RMSpropGrave())
                newton_tag.append(False)
            elif m == "FIRE":
                optimizer_instances.append(FIRE())
                newton_tag.append(False)
            elif m == "Adaderivative":
                optimizer_instances.append(Adaderivative())
                newton_tag.append(False)
            elif m == "CG" or m == "CG_PR" or m == "CG_FR" or m == "CG_HS" or m == "CG_DY":
                optimizer_instances.append(ConjgateGradient(method=m))
                newton_tag.append(False)
            # group of quasi-Newton method
               
            elif m == "BFGS" or m == "FSB" or m == "Bofill" or m == "MSP":
                optimizer_instances.append(Newton(method=m))
                optimizer_instances[i].DELTA = 0.10
                newton_tag.append(True)
            elif m == "mBFGS" or m == "mFSB" or m == "mBofill" or m == "mMSP":
                optimizer_instances.append(Newton(method=m))
                optimizer_instances[i].DELTA = 0.10
                newton_tag.append(True)
            elif m == "BFGS_LS" or m == "FSB_LS" or m == "Bofill_LS" or m == "MSP_LS":
                optimizer_instances.append(Newton(method=m))
                optimizer_instances[i].DELTA = 0.5
                optimizer_instances[i].linesearchflag = True
                newton_tag.append(True)
            elif m == "RFO_BFGS" or m == "RFO_FSB" or m == "RFO_Bofill" or m == "RFO_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order))
                optimizer_instances[i].DELTA = 0.50
                newton_tag.append(True)
            elif m == "RFO2_BFGS" or m == "RFO2_FSB" or m == "RFO2_Bofill" or m == "RFO2_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order))
                optimizer_instances[i].DELTA = 0.50
                newton_tag.append(True)
            elif m == "RFO3_BFGS" or m == "RFO3_FSB" or m == "RFO3_Bofill" or m == "RFO3_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order))
                optimizer_instances[i].DELTA = 0.50
                newton_tag.append(True)
            elif m == "mRFO_BFGS" or m == "mRFO_FSB" or m == "mRFO_Bofill" or m == "mRFO_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m, saddle_order=self.saddle_order))
                optimizer_instances[i].DELTA = 0.30
                newton_tag.append(True)
            else:
                
                print("This method is not implemented. :", m, " Thus, Default method is used.")
                optimizer_instances.append(Adabelief())
                newton_tag.append(False)
                
        self.method = method
        self.newton_tag = newton_tag
        return optimizer_instances
        
    def update_trust_radii(self, trust_radii, B_e, pre_B_e, pre_B_g, pre_move_vector):
    
        if self.trust_radii_update == "trust":
            Sc = 2.0
            Ce = (np.dot(pre_B_g.reshape(1, len(self.geom_num_list)*3), pre_move_vector.reshape(len(self.geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot(pre_move_vector.reshape(1, len(self.geom_num_list)*3), self.model_hess), pre_move_vector.reshape(len(self.geom_num_list)*3, 1)))
            r = (B_e - pre_B_e) / Ce
            r_min = 0.75
            r_good = 0.8
            if r <= r_min or r >= (2.0 - r_min):
                trust_radii /= Sc
                print("decrease trust radii")
            
            elif r >= r_good and r <= (2.0 -r_good):
                trust_radii *= Sc ** 0.5
                print("increase trust radii")
            else:
                print("keep trust radii")
                
        elif self.trust_radii_update == "legacy":
            if pre_B_e >= B_e:
                trust_radii *= 3.0
            else:
                trust_radii *= 0.1
        else:
            pass
                                   
        return np.clip(trust_radii, 0.1, 1.0)


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
        #update trust radii
        #-------------------------------------------------------------
        if self.iter % self.FC_COUNT == 0 and self.FC_COUNT != -1:
            self.trust_radii = 0.1
        elif self.FC_COUNT == -1:
            self.trust_radii = 1.0
        else:
            #self.model_hess = optimizer_instances[0].hessian + optimizer_instances[0].bias_hessian

            self.trust_radii = self.update_trust_radii(self.trust_radii, B_e, pre_B_e, pre_B_g, pre_move_vector)
            if self.saddle_order > 0:
                self.trust_radii = min(self.trust_radii, 0.1)
            
        #---------------------------------
        #calculate move vector
        #---------------------------------
        for i in range(len(optimizer_instances)):
            move_vector_list.append(optimizer_instances[i].run(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g))
            
            
        #---------------------------------
        # switch step update method
        #---------------------------------
        if len(move_vector_list) > 1:
            if abs(np.sqrt(np.square(B_g).mean())) > self.MAX_RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method:", self.method[0])
                #if self.newton_tag[0]:
                #    self.diag_hess_and_display(optimizer_instances[0])
            elif abs(np.sqrt(np.square(B_g).mean())) <= self.MAX_RMS_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) > self.MIN_RMS_FORCE_SWITCHING_THRESHOLD: 
                x_i = abs(np.sqrt(np.square(B_g).mean()))
                x_max = self.MAX_RMS_FORCE_SWITCHING_THRESHOLD
                x_min = self.MIN_RMS_FORCE_SWITCHING_THRESHOLD

                x_j = (x_i - x_min) / (x_max - x_min)
                
                f_val = 1 / (1 + np.exp(-10.0 * (x_j - 0.5)))
                move_vector = np.array(move_vector_list[0], dtype="float64") * f_val + np.array(move_vector_list[1], dtype="float64") * (1.0 - f_val)
                print(f_val, x_j)
                #if self.newton_tag[0]:
                #    self.diag_hess_and_display(optimizer_instances[0])
                    
                #if self.newton_tag[1]:
                #    self.diag_hess_and_display(optimizer_instances[1])
            
            else:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method:", self.method[1])
                #if self.newton_tag[1]:
                #    self.diag_hess_and_display(optimizer_instances[1])
                 
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
        return new_geometry, np.array(move_vector, dtype="float64"), optimizer_instances, self.trust_radii


