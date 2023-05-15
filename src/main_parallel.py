# Import modules
import numpy as np
import os

# Import functions
from common.assembly_A_local_global_matrices import *
from common.assembly_K_matrices import *
from common.assembly_B_matrices import *
from common.assembly_f_vectors import *
from common.assembly_u_solution import *
from common.RegularSudomainsMesh import RegularSubdomainsMesh
from common.utils import *

# Get the path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths based on the current file's location
d_path = os.path.join(current_dir, "..", "data", "small", "d.dat")
fP_path = os.path.join(current_dir, "..", "data", "small", "fP.dat")
fr_path = os.path.join(current_dir, "..", "data", "small", "fr.dat")
Ks_path = os.path.join(current_dir, "..", "data", "small", "localK.dat")
sol_path = os.path.join(current_dir, "..", "data", "small", "solution.dat")

# Import the data using the relative paths
d_dat = np.genfromtxt(d_path)
fP_dat = np.genfromtxt(fP_path)
fr_dat = np.genfromtxt(fr_path)
Ks = np.genfromtxt(Ks_path)
solution = np.genfromtxt(sol_path)
