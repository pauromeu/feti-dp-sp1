# Import modules
import numpy as np

# Import functions
from assembly_A_local_global_matrices import *
from assembly_K_matrices import *
from assembly_B_matrices import *
from assembly_f_vectors import *
from assembly_u_solution import *
from RegularSudomainsMesh import RegularSubdomainsMesh
from utils import *

# Import data
d_dat = np.genfromtxt('data/d.dat')
fP_dat = np.genfromtxt('data/fP.dat')
fr_dat = np.genfromtxt('data/fr.dat')
Ks = np.genfromtxt('data/localK.dat')
solution = np.genfromtxt('data/solution.dat')
