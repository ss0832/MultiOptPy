from .hessian_update import ModelHessianUpdate
import numpy as np
import copy
"""
RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
 Theor chim Acta (1992) 82: 189-205
"""

bohr2angstroms = 0.52917721067

class HybridCoordinateAugmentedRFO:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = self.config["saddle_order"] #
        self.element_list = self.config["element_list"] #
        self.natom = len(self.element_list) #
        self.iter = 0 #
        self.beta = 0.10
        self.covalent_radii_scale = 10.0
        self.initilization = True
        self.hybrid_hessian = None
        self.prev_hybrid_B_g = None
        self.prev_hybrid_g = None
        self.prev_hybrid_displacement = None
        self.bond_connectivity = None
        self.nonlinear_iter = 1000
        return
    
    def set_hessian(self, hessian):
        self.hessian = hessian
      
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    
    def hessian_update(self, displacement, delta_grad, hess):
        if "msp" in self.config["method"].lower():
            print("RFO_MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(hess, displacement, delta_grad)
        elif "bfgs" in self.config["method"].lower():
            print("RFO_BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(hess, displacement, delta_grad)
        elif "fsb" in self.config["method"].lower():
            print("RFO_FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(hess, displacement, delta_grad)
        elif "bofill" in self.config["method"].lower():
            print("RFO_Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(hess, displacement, delta_grad)
        else:
            raise "method error"
        return delta_hess
    
    
    def build_bond_connectivity(self, connect_matrix):
        atom_indices = np.transpose(np.triu(connect_matrix, k=1).nonzero())
        bonds = []
        for i, j in atom_indices:
            bonds.append([i, j])
        return bonds
    
    def bond_distance_matrix(self):
        radii = np.array([covalent_radii_lib(element) for element in self.element_list])
        return radii[:, np.newaxis] + radii[np.newaxis, :]

    def save_bond_connectivity(self, geom_num_list):
        tmp_geom_num_list = geom_num_list.reshape(self.natom, 3)
        diffs = tmp_geom_num_list[:, np.newaxis, :] - tmp_geom_num_list[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        bond_distances = self.bond_distance_matrix()
        bond_mask = (distances <= bond_distances * self.covalent_radii_scale) & (distances > 0) 
        bonds = self.build_bond_connectivity(bond_mask)
        return bonds

    def hybrid(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        # use hybrid coordinate which contains cartesian coordinate and internal coordinate
        print("Hybrid-coordinate-augmented RFO method")     
        if self.initilization:
            self.initilization = False
            self.bond_connectivity = self.save_bond_connectivity(geom_num_list)
            return self.DELTA*B_g
        

        print("saddle order:", self.saddle_order)
        
        # calculate B matrix and G matrix
        prim_B_mat = calc_B_mat(geom_num_list, self.bond_connectivity)
        prim_G_mat = calc_G_mat(prim_B_mat)
        B_mat = calc_delocalized_B_mat(prim_B_mat, prim_G_mat)
        G_mat = calc_G_mat(B_mat)
        inv_B_mat = calc_inv_B_mat(B_mat, G_mat)
        ncart = len(geom_num_list)
        nint = len(G_mat)
        n_tot = ncart + nint
        if self.iter == 0:
            self.int_hessian = np.ones((nint, nint))
            
        # calculate hybrid-coordinate geometry and gradient
        int_g = np.dot(inv_B_mat, g)
        int_pre_g = np.dot(inv_B_mat, pre_g)
        int_B_g = np.dot(inv_B_mat, B_g)
        int_pre_B_g = np.dot(inv_B_mat, pre_B_g)
        int_pre_geom = np.dot(B_mat, pre_geom)
        int_geom_num_list = np.dot(B_mat, geom_num_list)
       
        int_delta_grad = int_B_g.reshape(nint, 1) - int_pre_B_g.reshape(nint, 1)
        int_displacement = (int_geom_num_list - int_pre_geom).reshape(nint, 1)
        
        DELTA_for_QNM = self.DELTA
    
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            int_delta_hess = self.hessian_update(int_displacement, int_delta_grad, self.int_hessian)
            cart_delta_hess = self.hessian_update(geom_num_list, g, self.hessian)
            new_hess = self.hessian + self.bias_hessian + cart_delta_hess
            new_int_hess = self.int_hessian + int_delta_hess
        else:
            new_hess = self.hessian + self.bias_hessian
            
        matrix_for_RFO = np.append(new_hess, B_g, axis=1)
        tmp = np.array([np.append(B_g.T, 0.0)], dtype="float64")
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        print("lambda    (cart.) : ",lambda_for_calc)
        print("step size (cart.) : ",DELTA_for_QNM)

        cart_move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(ncart)), B_g)
       
        matrix_for_RFO = np.append(new_int_hess, int_B_g, axis=1)
        tmp = np.array([np.append(int_B_g.T, 0.0)], dtype="float64")
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        print("lambda    (int.) : ",lambda_for_calc)
        print("step size (int.) : ",DELTA_for_QNM)
        int_move_vector = DELTA_for_QNM * np.linalg.solve(new_int_hess - 0.1*lambda_for_calc*(np.eye(nint)), int_B_g)
        
       
        self.int_hessian += int_delta_hess
        self.iter += 1
        self.hessian += cart_delta_hess  
        
        # fitting hybrid_move_vector to the only cartesian coordinate
        ric_geom_num_list = np.dot(B_mat, geom_num_list)
        q_tgt = int_move_vector + ric_geom_num_list
        tmp_geom_num_list = copy.copy(geom_num_list)
        
        RRIC_move_vector = np.zeros((len(geom_num_list), 1))
        ric_diff = int_move_vector
        
        for jiter in range(self.nonlinear_iter):
            cart_diff = np.dot(inv_B_mat.T, ric_diff)
            tmp_geom_num_list = tmp_geom_num_list + cart_diff
            current_ric_geom_num_list = np.dot(B_mat, tmp_geom_num_list)
            ric_diff = q_tgt - current_ric_geom_num_list
            RRIC_move_vector += cart_diff
            if np.linalg.norm(cart_diff) < 1e-10:
                print("The nonlinear iteration is converged!!!: ITR.", jiter)
                break 
        else:
            print("Warning: The nonlinear iteration is not converged!!!")
        
        
        move_vec = -cart_move_vector / 2 + RRIC_move_vector / 2
        return move_vec


    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        move_vector = self.hybrid(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector
 

def calc_B_mat_int(geometry, bond_connectivity):#geometry: cartesian coordinate (3N, 1), bond_connectivity: bond connectivity matrix (M, 2)
    int_B_mat = np.zeros((len(bond_connectivity), len(geometry)))
    for i, bond in enumerate(bond_connectivity):
        i_idx = bond[0]
        j_idx = bond[1]
        
        ij_vec = geometry[3*i_idx:3*i_idx+2] - geometry[3*j_idx:3*j_idx+2]
        norm = np.linalg.norm(ij_vec)
        
        dr_dxi = (geometry[3*i_idx] - geometry[3*j_idx]) / norm  
        dr_dyi = (geometry[3*i_idx+1] - geometry[3*j_idx+1]) / norm  
        dr_dzi = (geometry[3*i_idx+2] - geometry[3*j_idx+2]) / norm  
        
        dr_dxj = -dr_dxi
        dr_dyj = -dr_dyi
        dr_dzj = -dr_dzi
        int_B_mat[i, 3*i_idx] = float(dr_dxi)
        int_B_mat[i, 3*i_idx+1] = float(dr_dyi)
        int_B_mat[i, 3*i_idx+2] = float(dr_dzi)
        int_B_mat[i, 3*j_idx] = float(dr_dxj)
        int_B_mat[i, 3*j_idx+1] = float(dr_dyj)
        int_B_mat[i, 3*j_idx+2] = float(dr_dzj)
    
    B_mat = int_B_mat
    return B_mat # (3N+M, 3N)


def calc_B_mat(geometry, bond_connectivity):#geometry: cartesian coordinate (3N, 1), bond_connectivity: bond connectivity matrix (M, 2)
    cart_B_mat = np.ones((len(geometry), len(geometry)))
    int_B_mat = np.zeros((len(bond_connectivity), len(geometry)))
    
    for i, bond in enumerate(bond_connectivity):
        i_idx = bond[0]
        j_idx = bond[1]
        
        ij_vec = geometry[3*i_idx:3*i_idx+2] - geometry[3*j_idx:3*j_idx+2]
        norm = np.linalg.norm(ij_vec)
        
        dr_dxi = (geometry[3*i_idx] - geometry[3*j_idx]) / norm  
        dr_dyi = (geometry[3*i_idx+1] - geometry[3*j_idx+1]) / norm  
        dr_dzi = (geometry[3*i_idx+2] - geometry[3*j_idx+2]) / norm  
        
        dr_dxj = -dr_dxi
        dr_dyj = -dr_dyi
        dr_dzj = -dr_dzi
        int_B_mat[i, 3*i_idx] = float(dr_dxi)
        int_B_mat[i, 3*i_idx+1] = float(dr_dyi)
        int_B_mat[i, 3*i_idx+2] = float(dr_dzi)
        int_B_mat[i, 3*j_idx] = float(dr_dxj)
        int_B_mat[i, 3*j_idx+1] = float(dr_dyj)
        int_B_mat[i, 3*j_idx+2] = float(dr_dzj)
    
    B_mat = np.append(cart_B_mat, int_B_mat, axis=0)
        
    return B_mat # (3N+M, 3N)


def calc_G_mat(B_mat):
    G_mat = np.dot(B_mat, B_mat.T)
    return G_mat

def calc_delocalized_B_mat(B_mat, G_mat):
    #ref.:J. Chem. Phys. 105, 192–212 (1996)
    Geigenval, Geigenvec = np.linalg.eigh(G_mat)
    
    nonzero_indices = np.where(np.abs(Geigenval) > 1e-10)[0]
    nonzero_eigvec = Geigenvec[:, nonzero_indices]
    delocalized_B_mat = np.dot(nonzero_eigvec.T, B_mat)
    
    return delocalized_B_mat
    
    

def calc_inv_B_mat(B_mat, G_mat):
    inv_B_mat = np.dot(np.linalg.pinv(G_mat), B_mat)
    return inv_B_mat


def covalent_radii_lib(element):#single bond
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, 
           "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, 
           "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, 
           "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,
           "Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42, 'X':1.000}#ang.
    # ref. Pekka Pyykkö; Michiko Atsumi (2009). “Molecular single-bond covalent radii for elements 1 - 118”. Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. (H...Rn)
            
    return CRL[element] / bohr2angstroms#Bohr

def number_element(num):
    elem = {1: "H",  2:"He",
         3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
        11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
        19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr",
        37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",43:"Tc",44:"Ru",45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe",
        55:"Cs", 56:"Ba", 57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy" ,67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn"}
        
    return elem[num]