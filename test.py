import numpy as np
import qutip as qt
import scipy as sp
from scipy import linalg
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as pltgs
import krotov
import os.path
import random
from matplotlib import rc
from cycler import cycler
import time
import pandas as pd
from Hamiltonian_library import *
from Ps_library import *
from config import *
rc('font',**{'family':'serif','serif':['Computer Modern'], 'size':25})
rc('text', usetex=True)

data_handler = handler()

N_atoms = int(1e5)
system = Ps_system(N_atoms)
system.init_MBdistribution()
system.init_states_desymmetrized()

N_pulses = 30
wavevector = 1
for i in range(N_pulses): # change number of pulses
    flip_pulse = {"rabi0" : 2*np.pi*(1000e-3), "detuning": 0,"chirp" : 2*np.pi*(50e-3),"pulse_duration" : 10,"unit_wavevector":wavevector,"start":i*30,"end":30*(1+i),"notch":1e-6}
    flip_pulse["label"] = "laser"+str(i)
    system.init_pulse(flip_pulse)
    wavevector = -wavevector
system.init_pulse_cycle()

system.set_Hamiltonian_notched_MT3()
system.evolve()

data_handler.save_states(system,"./data/states_notched/onetrain_30pulses.csv")