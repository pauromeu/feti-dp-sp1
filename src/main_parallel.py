# Import modules
import numpy as np

# Import functions
from common.assembly_A_local_global_matrices import *
from common.assembly_K_matrices import *
from common.assembly_B_matrices import *
from common.assembly_f_vectors import *
from common.assembly_u_solution import *
from common.RegularSudomainsMesh import RegularSubdomainsMesh
from common.utils import *

# Import data
d_dat = np.genfromtxt('../data/small/d.dat')
fP_dat = np.genfromtxt('../data/small/fP.dat')
fr_dat = np.genfromtxt('../data/small/fr.dat')
Ks = np.genfromtxt('../data/small/localK.dat')
solution = np.genfromtxt('../data/small/solution.dat')

print('done!')
