import numpy as np
import glob

from calc_tools import Calculationtools
from fileio import FileIO
from visualization import Graph

class ReactionPathRicciCurvature:
    def __init__(self, three_jacobian_mat, three_geodesic_dist_mat, three_euclidean_dist_mat):
        #doi:10.1002/jcc.27030
        self.jacob_mat = three_jacobian_mat
        self.geod_mat = three_geodesic_dist_mat
        self.eucl_mat = three_euclidean_dist_mat
        return
    

    
    def calc_Riemann_metric(self, k, i, j):
        tmp_mat = self.jacob_mat[k].T
        g_list = tmp_mat[i] * tmp_mat[j]
        
        #g_list = [self.jacob_mat[k][n][i] * self.jacob_mat[k][n][j] for n in range(len(self.jacob_mat[k]))]
        
        return np.sum(g_list)
    
    def calc_Riemann_inv_metric(self, k, i, j):
        tmp_mat = self.jacob_mat[k].T
        g_inv_list = 1 / tmp_mat[i] * 1 / tmp_mat[j]
        
        #g_list = [self.jacob_mat[k][n][i] * self.jacob_mat[k][n][j] for n in range(len(self.jacob_mat[k]))]
        
        return np.sum(g_inv_list)
    
    def calc_Riemann_1st_derivative_metric(self, i, j, a):
        dg_dt = 0.0
        if i == j and j == a:
            dg_dt = (self.calc_Riemann_metric(2, i, j) - self.calc_Riemann_metric(0, i, j)) / (self.geod_mat[2][a] - self.geod_mat[0][a])
        elif a != i and a != j:
            pass
        elif a == i and a != j:
            
            #dg_dt_list = [((self.jacob_mat[2][n][a] - self.jacob_mat[0][n][a]) * self.jacob_mat[1][n][j]) / (self.geod_mat[2][a] - self.geod_mat[0][a]) for n in range(len(self.jacob_mat[1]))]     
            dg_dt_list = ((self.jacob_mat[2].T[a] - self.jacob_mat[0].T[a]) * self.jacob_mat[1].T[j]) / (self.geod_mat[2][a] - self.geod_mat[0][a])
            dg_dt = np.sum(dg_dt_list) 
        
        else:
            #dg_dt_list = [((self.jacob_mat[2][n][a] - self.jacob_mat[0][n][a]) * self.jacob_mat[1][n][i]) / (self.geod_mat[2][a] - self.geod_mat[0][a]) for n in range(len(self.jacob_mat[1]))]
            dg_dt_list = ((self.jacob_mat[2].T[a] - self.jacob_mat[0].T[a]) * self.jacob_mat[1].T[i]) / (self.geod_mat[2][a] - self.geod_mat[0][a])
            dg_dt = np.sum(dg_dt_list) 
        
        return dg_dt
    
    def calc_Riemann_2nd_derivative_metric(self, i, j, a, b):
        d2g_dt2 = 0.0
        if (b == i and b == j) or (a == i and a == j):
            pass
        
        elif a == b and b == i and i == j:
            d2g_dt2 = ((self.calc_Riemann_metric(2, i, j) + self.calc_Riemann_metric(0, i, j))) / ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a]))
        
        elif a == b and b == i and b != j:
            #d2g_dt2_list = [((self.jacob_mat[2][n][a] - self.jacob_mat[0][n][a]) * self.jacob_mat[1][n][j]) / ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a])) for n in range(len(self.jacob_mat[1]))]
            d2g_dt2_list = ((self.jacob_mat[2].T[a] - self.jacob_mat[0].T[a]) * self.jacob_mat[1].T[j]) / ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a]))
            d2g_dt2 = np.sum(d2g_dt2_list)
            
        elif i == b and b != j and a == j:
            #d2g_dt2_list = [(((self.jacob_mat[2][n][b] - self.jacob_mat[1][n][b]) * self.jacob_mat[1][n][a]) - ((self.jacob_mat[2][n][b] - self.jacob_mat[1][n][b]) * self.jacob_mat[0][n][a])) / ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a])) for n in range(len(self.jacob_mat[1]))]
            d2g_dt2_list = (((self.jacob_mat[2].T[b] - self.jacob_mat[1].T[b]) * self.jacob_mat[1].T[a]) - ((self.jacob_mat[2].T[b] - self.jacob_mat[1].T[b]) * self.jacob_mat[0].T[a])) / ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a]))
            d2g_dt2 = np.sum(d2g_dt2_list)
             
        elif b != i and a == b and b == j:
            d2g_dt2_list = ((self.jacob_mat[2].T[a] - self.jacob_mat[0].T[a]) * self.jacob_mat[1].T[i]) / ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a])) 
            d2g_dt2 = np.sum(d2g_dt2_list)
        else:
            #d2g_dt2_list = [(((self.jacob_mat[2][n][b] - self.jacob_mat[1][n][b]) * self.jacob_mat[1][n][a]) - ((self.jacob_mat[2][n][b] - self.jacob_mat[1][n][b]) * self.jacob_mat[0][n][a]))/ ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a])) for n in range(len(self.jacob_mat[1]))]
            d2g_dt2_list = (((self.jacob_mat[2].T[b] - self.jacob_mat[1].T[b]) * self.jacob_mat[1].T[a]) - ((self.jacob_mat[2].T[b] - self.jacob_mat[1].T[b]) * self.jacob_mat[0].T[a]))/ ((self.geod_mat[2][a] - self.geod_mat[1][a]) * (self.geod_mat[1][a] - self.geod_mat[0][a])) 
                
            d2g_dt2 = np.sum(d2g_dt2_list)
        return d2g_dt2
    


    def calc_Riemann_inv_1st_derivative_metric(self, i, j, a):
        #dg_inv_dt_list = [((1/self.jacob_mat[2][n][i]) * (1/self.jacob_mat[2][n][j]) - (1/self.jacob_mat[0][n][i]) * (1/self.jacob_mat[0][n][j])) / (self.geod_mat[2][a] - self.geod_mat[0][a]) for n in range(len(self.jacob_mat[1]))]
        dg_inv_dt_list = ((1/self.jacob_mat[2].T[i]) * (1/self.jacob_mat[2].T[j]) - (1/self.jacob_mat[0].T[i]) * (1/self.jacob_mat[0].T[j])) / (self.geod_mat[2][a] - self.geod_mat[0][a]) 
        
        return np.sum(dg_inv_dt_list)


    def calc_Christoffel_symbol(self, a, b, c):#Γ_a_bc
        #tmp_array = np.arange(0, len(self.jacob_mat[0]), 1)
        #a_array = np.ones(len(self.jacob_mat[0])) * a
        #b_array = np.ones(len(self.jacob_mat[0])) * b
        #c_array = np.ones(len(self.jacob_mat[0])) * c
        #one_array = np.ones(len(self.jacob_mat[0]))
        
        christoffel_symbol_list = [0.5 * (self.calc_Riemann_inv_metric(1, a, i) * (self.calc_Riemann_1st_derivative_metric(i, b, c) + self.calc_Riemann_1st_derivative_metric(i, c, b) - self.calc_Riemann_1st_derivative_metric(b, c, i))) for i in range(len(self.jacob_mat[0]))]
        #christoffel_symbol_list = 0.5 * (self.calc_Riemann_inv_metric(one_array, a_array, tmp_array) * (self.calc_Riemann_1st_derivative_metric(tmp_array, b_array, c_array) + self.calc_Riemann_1st_derivative_metric(tmp_array, c_array, b_array) - self.calc_Riemann_1st_derivative_metric(b_array, c_array, tmp_array)))
        
        return sum(christoffel_symbol_list)
    
    def calc_1st_derivative_Christoffel_symbol(self, a, b, c, d):#dΓ_a_bc/dζ_d
        first_derivative_christoffel_symbol_list = [0.5 * (self.calc_Riemann_inv_1st_derivative_metric(a, i, d) * (self.calc_Riemann_1st_derivative_metric(i, b, c) + self.calc_Riemann_1st_derivative_metric(i, c, b) - self.calc_Riemann_1st_derivative_metric(b, c, i)) + self.calc_Riemann_inv_metric(1, a, i) * (self.calc_Riemann_2nd_derivative_metric(i, b, c, d) + self.calc_Riemann_2nd_derivative_metric(i, c, b, d) - self.calc_Riemann_2nd_derivative_metric(b, c, i, d))) for i in range(len(self.jacob_mat[0]))]
            
            
        return sum(first_derivative_christoffel_symbol_list)
    
    def calc_Riemann_curvature_tensor(self, a, b, c, d):#R_a_bcd
        riemann_curvature = self.calc_1st_derivative_Christoffel_symbol(a, c, d, b) - self.calc_1st_derivative_Christoffel_symbol(a, b, d, c)
        
        tmp_list = [self.calc_Christoffel_symbol(i, c, d) * self.calc_Christoffel_symbol(a, b, i) -1 * self.calc_Christoffel_symbol(i, b, d) * self.calc_Christoffel_symbol(a, c, i) for i in range(len(self.jacob_mat[0]))]
        riemann_curvature += sum(tmp_list)
        return riemann_curvature 
    
    def calc_Ricci_curvature(self):
        size = len(self.jacob_mat[0])
        ricci_scalar = sum(self.calc_Riemann_inv_metric(1, i, j) * self.calc_Riemann_curvature_tensor(n, i, n, j) for n in range(size) for i in range(size) for j in range(size))
        return ricci_scalar

#------------
#This implementation must be wrong.
#------------
class CalculationCurvature:
    def __init__(self, file_directory):
        self.BPA_FOLDER_DIRECTORY = file_directory
        self.dummy_args_electric_charge_and_multiplicity = [0, 1]
        return
        
    def main(self):
        print("Calculate Ricci scalar of calculated aprrox. reaction path.")
        file_list = sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz"))
        step_num = len(file_list)
        
        rxn_path_ricci_curvature_list = []
        num_list = []
        

        
        for i in range(2, step_num - 3):
            print("# NODE", i)
            mFIO = FileIO(self.BPA_FOLDER_DIRECTORY, file_list[i-1])
            FIO = FileIO(self.BPA_FOLDER_DIRECTORY, file_list[i])
            pFIO = FileIO(self.BPA_FOLDER_DIRECTORY, file_list[i+1])
            p2FIO = FileIO(self.BPA_FOLDER_DIRECTORY, file_list[i+2])
            m_geometry_list, element_list, _ = mFIO.make_geometry_list(self.dummy_args_electric_charge_and_multiplicity)
            geometry_list, element_list, _ = FIO.make_geometry_list(self.dummy_args_electric_charge_and_multiplicity)
            p_geometry_list, element_list, _ = pFIO.make_geometry_list(self.dummy_args_electric_charge_and_multiplicity)
            p2_geometry_list, element_list, _ = p2FIO.make_geometry_list(self.dummy_args_electric_charge_and_multiplicity)
            m_geom_num_list = []
            geom_num_list = []
            p_geom_num_list = []
            p2_geom_num_list = []
           
            for j in range(len(element_list)):
                m_geom_num_list.append(m_geometry_list[0][j+1][1:4])
                geom_num_list.append(geometry_list[0][j+1][1:4])
                p_geom_num_list.append(p_geometry_list[0][j+1][1:4])
                p2_geom_num_list.append(p2_geometry_list[0][j+1][1:4])
            p_geom_num_list = np.array(p_geom_num_list, dtype="float64")    
            p2_geom_num_list = np.array(p2_geom_num_list, dtype="float64")    
            geom_num_list = np.array(geom_num_list, dtype="float64")    
            m_geom_num_list = np.array(m_geom_num_list, dtype="float64")
            
            if i == 2:
                base_geodesic_dist_mat = geom_num_list.reshape(len(element_list)*3, 1) * 0.0
                base_euclidean_mat = geom_num_list.reshape(len(element_list)*3, 1) * 0.0
            
            m_geodesic_dist_mat = Calculationtools().calc_geodesic_distance(m_geom_num_list, geom_num_list).reshape(len(element_list)*3, 1) + base_geodesic_dist_mat
            geodesic_dist_mat = Calculationtools().calc_geodesic_distance(geom_num_list, p_geom_num_list).reshape(len(element_list)*3, 1) + m_geodesic_dist_mat
            p_geodesic_dist_mat = Calculationtools().calc_geodesic_distance(p_geom_num_list, p2_geom_num_list).reshape(len(element_list)*3, 1) + geodesic_dist_mat
            
            m_euclidean_dist_mat = Calculationtools().calc_euclidean_distance(m_geom_num_list, geom_num_list).reshape(len(element_list)*3, 1) + base_euclidean_mat
            euclidean_dist_mat = Calculationtools().calc_euclidean_distance(geom_num_list, p_geom_num_list).reshape(len(element_list)*3, 1) + m_euclidean_dist_mat
            p_euclidean_dist_mat = Calculationtools().calc_euclidean_distance(p_geom_num_list, p2_geom_num_list).reshape(len(element_list)*3, 1) + euclidean_dist_mat
            
            m_jacob_mat = Calculationtools().gen_n_dinensional_rot_matrix(m_geodesic_dist_mat, m_euclidean_dist_mat)
            jacob_mat = Calculationtools().gen_n_dinensional_rot_matrix(geodesic_dist_mat, euclidean_dist_mat)
            p_jacob_mat = Calculationtools().gen_n_dinensional_rot_matrix(p_geodesic_dist_mat, p_euclidean_dist_mat)
            
            three_geodesic_dist_mat = [m_geodesic_dist_mat, geodesic_dist_mat, p_geodesic_dist_mat]
            three_euclidean_dist_mat = [m_euclidean_dist_mat, euclidean_dist_mat, p_euclidean_dist_mat]
            three_jacobian_mat = [m_jacob_mat, jacob_mat, p_jacob_mat]
            
            RPRC = ReactionPathRicciCurvature(three_jacobian_mat, three_geodesic_dist_mat, three_euclidean_dist_mat)
            
            ricci_curvature = RPRC.calc_Ricci_curvature()
            print("Ricci curvature:", ricci_curvature)
            rxn_path_ricci_curvature_list.append(ricci_curvature)
            num_list.append(i)
            base_geodesic_dist_mat = m_geodesic_dist_mat
            base_euclidean_mat = m_euclidean_dist_mat
        
        rxn_path_ricci_curvature_list = np.array(rxn_path_ricci_curvature_list, dtype="float64")
        num_list = np.array(num_list)
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.single_plot(num_list, rxn_path_ricci_curvature_list, "", 0, axis_name_1="ITR. ", axis_name_2="Ricci_curvature", name="Ricci_curvature_R")
        G.single_plot(num_list, np.log10(np.abs(rxn_path_ricci_curvature_list)), "", 0, axis_name_1="ITR. ", axis_name_2="Ricci_curvature (log10|R|)", name="Ricci_curvature_logR")
        
        with open(self.BPA_FOLDER_DIRECTORY+"ricci_curvature.csv", "w") as f:
            f.write("iter., Ricci curvature R\n")
            for i in range(len(num_list)):
                f.write(str(num_list[i])+","+str(float(rxn_path_ricci_curvature_list[i]))+"\n")
            
        return
    
    def main_for_neb(self, geometry_list):
        #TODO: implementation for neb method
        return