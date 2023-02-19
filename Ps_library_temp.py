import numpy as np
import qutip as qt
import scipy as sp
from scipy import linalg
import matplotlib
import matplotlib.pylab as plt
import krotov
import os.path
import random
from matplotlib import rc
from cycler import cycler
import time
import pandas as pd
import gc

from Hamiltonian_library import *
from config import *

class laser:
    def __init__(self,pulse_kwargs):
        # Initialising time-variables
        self.laser = pulse_kwargs["label"]
        self.startTime = pulse_kwargs["start"]
        self.endTime =  pulse_kwargs["end"] #ps 
        
        if "scale" in pulse_kwargs:
            self.binwidth = 2*np.pi/omega0*pulse_kwargs["scale"]
        else:
            self.binwidth = 2*np.pi/omega0*100
        self.N_time = int((self.endTime-self.startTime)/self.binwidth)
        self.tlist = np.linspace(self.startTime,self.endTime,self.N_time)
        self.tcentre = (self.endTime-self.startTime)/2 + self.startTime #ps
        self.tlist_centre = np.full(self.N_time,self.tcentre)
        
    
        self.detuning = pulse_kwargs["detuning"]
        self.chirp0 = pulse_kwargs["chirp"]
        self.rabi0= pulse_kwargs["rabi0"]
        self.pulse_duration = pulse_kwargs["pulse_duration"]
        self.omega_L0 = omega0 + self.detuning # central laser frequency
        self.label = pulse_kwargs["label"]
        self.direction = pulse_kwargs["unit_wavevector"]
        self.order = pulse_kwargs["order"]
        #self.dipole_moment = 1*Debye
        #self.beam_width = 5e-1 # cm
        #self.energy = np.pi**2*hbar_eV**2*c*eps0/(32*np.sqrt(2*np.log(2)))*self.rabi0**2*self.beam_width**3*self.pulse_duration/self.dipole_moment**2
        # (eV ps)^2 * (cm/ps) * (e^2 / eV / cm) * THz^2 * cm^3 * ps / (e cm)^2
        # eV    * cm
       

        
        self.wavenumber_value = omega0/c
        if "notch" in pulse_kwargs:
            self.notch0 = pulse_kwargs["notch"] # cm/ps
            self.notch_THz = self.notch0 /c*omega0 /(2*np.pi) # THz
        self.unit_wavevector = pulse_kwargs["unit_wavevector"]

        self.chirp = lambda tlist,args: [self.chirp0 * (t-self.tcentre)  if t >= self.startTime and t < self.endTime else 0 for t in tlist]
        self.wavenumber = lambda tlist,args: [self.unit_wavevector*self.wavenumber_value if t >= self.startTime and t < self.endTime else 0 for t in tlist] 
        self.wavevector = lambda tlist,args: [self.unit_wavevector if t >= self.startTime and t < self.endTime else 0 for t in tlist]
        self.rabi = lambda t, args: self.rabi0 * np.exp(-4*np.log(2)*(t-self.tcentre)**2/self.pulse_duration**2)
        
        # rabi = d*E/hbar => E = hbar*rabi/d
        self.rabi_beating = lambda t, args: 2*self.rabi0 * np.exp(-4*np.log(2)*(t-self.tcentre)**2/self.pulse_duration**2)*np.abs(np.sin(self.detuning*(t-self.tcentre)))
        self.rabi_beating2 = lambda t, args: self.rabi0 * np.exp(-4*np.log(2)*(t-self.tcentre)**2/self.pulse_duration**2)*2/np.pi*np.arctan(self.notch_THz*(t-self.tcentre))
        self.selector1 = np.full(self.N_time, 1 if self.unit_wavevector == -1 else 0)
        self.selector2 = np.full(self.N_time, 1 if self.unit_wavevector == 1 else 0)
        
# handles qutip data
# plotter as well
class handler:
    def __init__(self):
        self.figures = dict()
        return None
    
    def save_states_csv(self,obj,path):
        qt.qsave(obj.saved_states,path)
    
    def load_states_csv(self,path):
        return qt.qload(path)

    def figure_pulse(self,obj):
        return None
        #self.figures["pulse"] = (fig,axes)
        
    def get_std(self,obj):
        N_pulses = len(obj.saved_states)
        print(N_pulses)
        std = np.zeros(N_pulses)
        for j in range(N_pulses):
            N_g,N_e,N = obj.get_states(obj.saved_states[j].unit())
            std[j] = np.sqrt(np.sum([p_i*v_i**2 for (p_i,v_i) in [(N[i],obj.velocity_bins[i]) for i in range(obj.N_bins)] ]))
        return std


    def expect_standard_deviation(self,DM,N_bins):
        DM_vel = DM.ptrace(0).unit() # partial trace to get DM over velocity space
        oper_momentum = qt.num(N_bins,offset=-N_bins+1)
        #print("first moment: "+str((oper_momentum*DM_vel).tr()**2))
        #print("second moment: "+str((oper_momentum**2*DM_vel).tr()))
        std = np.sqrt((oper_momentum**2*DM_vel).tr()-((oper_momentum*DM_vel).tr())**2)
        return std


class Ps_system(HamiltonianClass):
    def __init__(self,N_atoms=1,N_points=250):
        self.T = 300 #K temperature of cloud
        self.m = 2*511e3 # eV/c^2
        self.std_deviation = np.sqrt(k*self.T/(self.m/c**2)) #standard deviation of gaussian
        self.amplitude = np.sqrt((self.m/c**2)/(2*np.pi*k*self.T))
        self.N_atoms = N_atoms
        #self.rate = 1
        
        self.N_points = N_points
        # if self.N_points % 2 == 0:
        #     print(True)
        #     self.N_points+=1
        self.momentum_bins = np.arange(-self.N_points/2,self.N_points/2)
        print(self.momentum_bins.shape)
        self.dv = 1.5e-7 # cm/ps, calculated such that 1 step is the equivalent of 1 unit of photon momentum for Ps        
        self.velocity_bins = self.dv*self.momentum_bins
        #self.velocity_bins = np.linspace(-self.max_vel,self.max_vel,self.N_bins) #cm/ps
        self.initial_pop = np.zeros(self.N_points)
    
        self.dp = 173 # eV ps/cm, 1 unit of momentum transfer with wavelength 247e-9nm onto Ps
        self.real_momentum_bins = self.dp*self.momentum_bins #eV ps/cm 
        self.max_vel = self.dv*self.N_points//2
        self.max_detuning = self.max_vel/c*omega0 /(2*np.pi) * 1e3 # GHz
        self.detuning_bins = np.linspace(-self.max_detuning,self.max_detuning,self.N_points)
        
        # Default values. Safe to override outside of class definition
        self.flag_SE_simple = False
        self.flag_SE_distributive = False
        self.flag_annihilation = False
        self.flag_photoionisation = False

        self.internal_dims = 2
        self.kets_vel = [qt.basis(self.N_points,n) for n in range(self.N_points)]
        
        self.c_ops = []
        self.e_ops = []
        self.laserDict = dict()
        self.H = []
        self.saved_states =[]
        #self.saved_expect = []

        self.idx_e_ops = dict(); self.order = 0
        self.expect = dict()
            
    
    def init_distribution_singular(self):
        self.initial_pop[self.N_points//2] = self.N_atoms
    def init_distribution_flattop(self):
        #self.initial_pop = np.full(self.N_points,self.N_atoms/self.N_points)
        self.initial_pop = np.full(self.N_points,1)
        
    
    # Assume Maxwell-Boltzmann distribution
    def init_MBdistribution(self,v0=0,std_deviation=None):
        if std_deviation == None:
            std_deviation = self.std_deviation
        for i in range(self.N_points-1):
            self.initial_pop[i] = sp.integrate.quad(lambda v: self.amplitude*np.exp(-(v-v0)**2/(2*std_deviation**2))*self.N_atoms,self.velocity_bins[i],self.velocity_bins[i+1])[0]    

    # instantiate a new object with name given by label
    def init_pulse(self,pulse_kwargs):
        laserLabel = pulse_kwargs["label"]
        laserObj = laser(pulse_kwargs)
        self.laserDict[laserLabel] = laserObj
    
    
    def init_pulse_cycle(self,notch=None):
        self.laserDict = sorted(self.laserDict.items(),key=lambda x:x[1].order)

        self.startTime = self.laserDict[0][1].startTime
        self.endTime =  self.laserDict[-1][1].endTime
        
        

        self.binwidth = 2*np.pi/omega0*100
        self.N_time = int(self.endTime/self.binwidth)
        self.tlist,self.dt = np.linspace(0,self.endTime,self.N_time,retstep=True)
        self.wavenumber_value = self.laserDict[0][1].wavenumber_value

        self.rabi = np.sum([laser[1].rabi(self.tlist,None) for laser in self.laserDict],axis=0)
        self.chirp = np.sum([laser[1].chirp(self.tlist,None) for laser in self.laserDict],axis=0)
        self.wavenumber = np.sum([laser[1].wavenumber(self.tlist,None) for laser in self.laserDict],axis=0)
        self.wavevector = np.sum([laser[1].wavevector(self.tlist,None) for laser in self.laserDict],axis=0)   

        # func which is 1 when wavevector is 1 and 0 everywhere else
        # -"- when wavevector is -1 and 0 everywhere else
        self.func1 = np.asarray([1 if k == -1 else 0 for k in self.wavevector])
        self.func2 = np.asarray([1 if k == 1 else 0 for k in self.wavevector])

        # functions for notched spectra
        #self.notch_function = np.sum([laser[1].notch(self.tlist,None) for laser in self.laserDict],axis=0)   
        self.rabi_beating = np.sum([laser[1].rabi_beating(self.tlist,None) for laser in self.laserDict],axis=0)
        self.rabi_beating2 = np.sum([laser[1].rabi_beating2(self.tlist,None) for laser in self.laserDict],axis=0)

    
    def init_states_general_flattop(self,internal_state_arr,ret=False):
        vel_DM = qt.qdiags(self.initial_pop,0) # density matrix    
        self.states = qt.tensor(vel_DM,qt.Qobj(internal_state_arr)) # density matrix, composite of g/e space and vel space

        if ret == True:
            return self.states


    # Initialise states in ground
    def init_states_ground(self,ret=False):
        vel_DM = qt.qdiags(self.initial_pop,0) # density matrix    
        DM_1S = qt.basis(self.internal_dims,0)
        self.states = qt.tensor(vel_DM,DM_1S) # density matrix, composite of g/e space and vel space

        if ret == True:
            return self.states


    # Initialise states in excited
    def init_states_excited(self,ret=False):
        vel_DM = qt.qdiags(self.initial_pop,0) # density matrix    
        DM_2P = qt.ket2dm(qt.basis(self.internal_dims,1))
        states = qt.tensor(vel_DM,DM_2P) # density matrix, composite of g/e space and vel space
        
        if ret == True:
            return states
        else:
            self.states = states


    # Initialise desymmetrized states
    def init_states_desymmetrized(self,ret=False):
        DM_1S = qt.ket2dm(qt.basis(self.internal_dims,0))
        DM_2P = qt.ket2dm(qt.basis(self.internal_dims,1))
        #vel_DM = qt.Qobj(np.diag(self.velocity_bins))
        #negatives = self.velocity_bins.copy() ; negatives[self.velocity_bins > 0] = 0 
        #positives = self.velocity_bins.copy() ; positives[self.velocity_bins < 0] = 0 
        

        #self.states = qt.tensor(qt.Qobj(np.diag(negatives)),DM_1S) + qt.tensor(qt.Qobj(np.diag(positives)),DM_2P)
        
        # ground = np.zeros(self.N_bins)
        # excited = np.zeros(self.N_bins)
        # for i in range(self.N_bins):
        #     if i < self.N_bins//2:
        #         ground[i] = np.sqrt(self.initial_pop[i])
        #     else:
        #         excited[i] = np.sqrt(self.initial_pop[i])

        # ground_DM = qt.Qobj(ground)*qt.Qobj(ground).dag() # density matrix    
        # excited_DM = qt.Qobj(excited)*qt.Qobj(excited).dag() # density matrix   
        
        ground_DM = qt.qdiags([self.initial_pop[i] if i < self.N_points//2 else 0 for i in range(self.N_points)],0)
        excited_DM = qt.qdiags([self.initial_pop[i] if i > self.N_points//2 else 0 for i in range(self.N_points)],0)
        
        self.states = qt.tensor(ground_DM,DM_1S) +qt.tensor(excited_DM,DM_2P) 
        
        
        if ret == True:
            return self.states


    def save_states(self):
        time = 0
        self.checkpoints = [time]
        for laser in self.laserDict:
            time = laser[1].endTime
            self.checkpoints.append(time)

        for time in self.checkpoints:
            idx = int(time/self.dt)
            self.saved_states.append( self.result.states[idx])  
    


    def organise_result_expect(self,result):
        # the class object keeps track of where the populations of each internal dimension is indexed in result.expect
        # for each internal dimension, make a new dictionary entry with label corresponding to internal dimension and 
        for label in self.idx_e_ops:         
            idx_range = self.idx_e_ops[label]

            # check to see if dict already has an entry with keyword given by label
            # if yes, append a new array to the existing value
            # each new iteration is then a new pulse
            if label in self.expect: 
                self.expect[label] = np.append(self.expect[label],[np.asarray(result.expect)[idx_range,-1]],axis=0)    
            else:
                self.expect[label] = [np.asarray(result.expect)[idx_range,0],np.asarray(result.expect)[idx_range,-1]]
            

    # interpret this DM as if it were a DM of this exact system, with the same e_ops and c_ops
    def interpret_state(self,DM):       
        expect = dict()
        for label in self.idx_e_ops:
            idx_range = self.idx_e_ops[label]
            #print(idx_range)
            for n in idx_range:
                if label in expect:
                    expect[label].append((self.e_ops[n]*DM).tr() )
                else:
                    expect[label] = [(self.e_ops[n]*DM).tr()]
                
        return expect


    def newfunc_saved_dm(self,result,save_path):
        # save current state to file, and then delete previous state from memory
        
        pass


    def evolve(self,save_path=None):
        self.laserDict = sorted(self.laserDict.items(),key=lambda x:x[1].order)
        opts = qt.Options(store_states=True)
        self.saved_states.append(self.states)
        for i in range(len(self.laserDict)):
            laser = self.laserDict[i][1]
            args = {"chirp":np.asarray(laser.chirp(laser.tlist,None)),
                    "wavevector":laser.direction,
                    "rabi":np.asarray(laser.rabi(laser.tlist,None)),
                    "beating":np.asarray(laser.rabi_beating2(laser.tlist,None)),
                    "selector1":laser.selector1,
                    "selector2":laser.selector2,
                    "tlist":laser.tlist,
                    "omega_L0":laser.omega_L0}
            
            self.create_composite(laser)
            self.set_Hamiltonian_MT(args)
            #print("direction of "+laser.label+": "+str(laser.direction))
            print("Simulating pulse number "+str(laser.order)+" ....")
            result = qt.mesolve(self.H, self.states,laser.tlist, e_ops=self.e_ops,c_ops=self.c_ops, options = opts)
            print("Done!")
            self.organise_result_expect(result)
            if save_path == None:
                self.saved_states.append(result.states[-1])
                self.states = result.states[-1]
            else:
                new_path = save_path +laser.label+".csv"
                self.states = result.states[-1]
                qt.qsave(self.states,new_path)
