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
from Hamiltonian_library import *
from config import *

class laser:
    def __init__(self,pulse_kwargs):
        # Initialising time-variables
        self.startTime = pulse_kwargs["start"]
        self.endTime =  pulse_kwargs["end"] #ps 
        
        self.binwidth = 2*np.pi/omega0*100
        self.N_time = int((self.endTime-self.startTime)/self.binwidth)
        self.tlist = np.linspace(self.startTime,self.endTime,self.N_time)
        self.tcentre = (self.endTime-self.startTime)/2 + self.startTime #ps
        self.tlist_centre = np.full(self.N_time,self.tcentre)
        
        self.detuning = pulse_kwargs["detuning"]
        self.chirp0 = pulse_kwargs["chirp"]
        self.rabi0= pulse_kwargs["rabi0"]
        self.pulse_duration = pulse_kwargs["pulse_duration"]
        
        self.dipole_moment = 1*Debye
        self.beam_width = 5e-1 # cm
        self.energy = np.pi**2*hbar_eV**2*c*eps0/(32*np.sqrt(2*np.log(2)))*self.rabi0**2*self.beam_width**3*self.pulse_duration/self.dipole_moment**2
        # (eV ps)^2 * (cm/ps) * (e^2 / eV / cm) * THz^2 * cm^3 * ps / (e cm)^2
        # eV    * cm
       
        self.label = pulse_kwargs["label"]
        self.wavenumber_value = omega0/c
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
    
    
    def jump_photoionisation(self):
        self.collapse_rate = np.sqrt(photoionisation_cross_section*eps0*c/(2*omega0)*self.rabi(self.tlist,None)**2*hbar_eV/Debye**2) # sqrt of the photoionisation rate
    
    
    def jump_spontaneous_emission(self):
        #self.collapse_rate = 1/3e3 # ps
        self.collapse_rate = 1/3 # ps
    
    
    def jump_annihilation(self):
        #self.collapse_rate = 1/142e3 # ps
        self.collapse_rate = 1/142 # ps
        
        
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

class Ps_system(HamiltonianClass):
    def __init__(self,N_atoms=1,isDissipative=False,dims=2):
        self.T = 300 #K temperature of cloud
        self.m = 2*511e3 # eV/c^2
        self.std_deviation = np.sqrt(k*self.T/(self.m/c**2)) #standard deviation of gaussian
        self.amplitude = np.sqrt((self.m/c**2)/(2*np.pi*k*self.T))
        self.N_atoms = N_atoms
        #self.rate = 1
        
        self.N_bins = 250
        self.dv = 1.5e-7 # cm/ps, calculated such that 1 step is the equivalent of 1 unit of photon momentum for Ps
        self.max_vel = self.N_bins*self.dv/2
        
        self.velocity_bins = np.linspace(-self.max_vel,self.max_vel,self.N_bins) #cm/ps
        self.initial_pop = np.zeros(self.N_bins)
        self.dv_bins = np.arange(-self.N_bins//2,self.N_bins//2)
        
    
        self.dp = 173 # eV ps/cm, 1 unit of momentum transfer with wavelength 247e-9nm onto Ps
        self.max_momentum = self.dp*self.N_bins//2 # eV ps/cm
        self.momentum_bins = np.linspace(-self.max_momentum,self.max_momentum,self.N_bins) #eV ps/cm 
        self.bins_bins = np.arange(-self.N_bins//2,self.N_bins//2) # array of bins
        
        self.max_detuning = self.max_vel/c*omega0 /(2*np.pi) * 1e3 # GHz
        self.detuning_bins = np.linspace(-self.max_detuning,self.max_detuning,self.N_bins)
        
        
        self.kets = [qt.basis(self.N_bins,n) for n in range(self.N_bins)]
        
        
        
        self.init_expect_oper(dims)

        self.laserDict = dict()
        self.H = []
        self.saved_states =[]
        self.saved_expect = []

        self.createTensors(isDissipative)
            
              
    def init_distribution_singular(self):
        self.initial_pop[self.N_bins//2] = self.N_atoms
    def init_distribution_constant(self):
        self.initial_pop = np.full(self.N_bins,self.N_atoms/self.N_bins)
    
    # Assume Maxwell-Boltzmann distribution
    def init_MBdistribution(self,v0=0,std_deviation=None):
        if std_deviation == None:
            std_deviation = self.std_deviation
        for i in range(self.N_bins-1):
            self.initial_pop[i] = int(sp.integrate.quad(lambda v: self.amplitude*np.exp(-(v-v0)**2/(2*std_deviation**2))*1e5,self.velocity_bins[i],self.velocity_bins[i+1])[0])

    def reset_distribution(self,states):
        N_g,N_e,N = self.get_states(states)
        self.initial_pop = N
        self.init_states_desymmetrized()
    # get initial Maxwell-Boltzmann distribution
    def get_initial_pop_plot(self):
        fig,ax = plt.subplots()
        #ax.fill_between(self.velocity_bins,0,self.initial_pop)
        ax.plot(self.velocity_bins,self.initial_pop,'.')
        ax.set_xlabel("velocities [cm/ps]")
        ax.set_ylabel("Number of Ps")
        
        return fig,ax
    

    def get_states(self, states="default"):
        ## for some reason the default states dont give the right result
        if states == "default":
            N_g = np.asarray([np.abs(self.states[2*i,2*i]) for i in range(self.N_bins)])
            N_e = np.asarray([np.abs(self.states[2*i+1,2*i+1]) for i in range(self.N_bins)])
            N = np.asarray([np.abs(self.states[2*i,2*i])+np.abs(self.states[2*i+1,2*i+1]) for i in range(self.N_bins)]) # total
        else:
            N_g = np.asarray([np.abs(states[2*i,2*i]) for i in range(self.N_bins)])
            N_e = np.asarray([np.abs(states[2*i+1,2*i+1]) for i in range(self.N_bins)])
            N = np.asarray([np.abs(states[2*i,2*i])+np.abs(states[2*i+1,2*i+1]) for i in range(self.N_bins)]) # total
        return N_g,N_e,N
    


    # instantiate a new object with name given by label
    def init_pulse(self,pulse_kwargs):
        laserLabel = pulse_kwargs["label"]
        laserObj = laser(pulse_kwargs)
        self.laserDict[laserLabel] = laserObj
    
    
    def init_pulse_cycle(self,notch=None):
        self.laserDict = sorted(self.laserDict.items(),key=lambda x:x[1].startTime)

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
        vel_DM = qt.Qobj(np.sqrt(self.initial_pop))*qt.Qobj(np.sqrt(self.initial_pop)).dag() # density matrix    
        self.states = qt.tensor(vel_DM,qt.Qobj(internal_state_arr)) # density matrix, composite of g/e space and vel space

        if ret == True:
            return self.states
    # Initialise states in ground
    def init_states_ground(self,ret=False):
        vel_DM = qt.Qobj(np.sqrt(self.initial_pop))*qt.Qobj(np.sqrt(self.initial_pop)).dag() # density matrix    
        self.states = qt.tensor(vel_DM,qt.Qobj([[1,0],[0,0]])) # density matrix, composite of g/e space and vel space

        if ret == True:
            return self.states

    # Initialise states in excited
    def init_states_excited(self,ret=False):
        vel_DM = qt.Qobj(np.sqrt(self.initial_pop))*qt.Qobj(np.sqrt(self.initial_pop)).dag() # density matrix    
        self.states = qt.tensor(vel_DM,qt.Qobj([[0,0],[0,1]])) # density matrix, composite of g/e space and vel space
        
        if ret == True:
            return self.states
        
    def return_plot_all_states(self):
        N_ground = self.get_ground()
        N_excited = self.get_excited()

        fig = plt.figure()
        plt.bar(self.velocity_bins,N_ground,width=self.dv,color="blue",label="N_g")
        plt.bar(self.velocity_bins,N_excited,width=self.dv,color="green",bottom=N_ground,label="N_e")
        plt.xlabel("velocities [cm/ps]")
        plt.ylabel("Number of atoms")
        return fig        

    # Initialise desymmetrized states
    def init_states_desymmetrized(self,ret=False):
        ground = np.zeros(self.N_bins)
        excited = np.zeros(self.N_bins)
        for i in range(self.N_bins):
            if i < self.N_bins//2:
                ground[i] = np.sqrt(self.initial_pop[i])
            else:
                excited[i] = np.sqrt(self.initial_pop[i])

        ground_DM = qt.Qobj(ground)*qt.Qobj(ground).dag() # density matrix    
        excited_DM = qt.Qobj(excited)*qt.Qobj(excited).dag() # density matrix   
        
        self.states = qt.tensor(ground_DM,qt.Qobj([[1,0],[0,0]])) +qt.tensor(excited_DM,qt.Qobj([[0,0],[0,1]])) 
        if ret == True:
            return self.states

    
    def init_states_desymmetrized2(self):
        ground = np.zeros(self.N_bins)
        excited = np.zeros(self.N_bins)
        for i in range(self.N_bins):
            if i < self.N_bins//2:
                if i % 2 == 0:
                    ground[i] = np.sqrt(self.initial_pop[i])
            else:
                if i % 2 != 0:
                    excited[i] = np.sqrt(self.initial_pop[i])

        ground_DM = qt.Qobj(ground)*qt.Qobj(ground).dag() # density matrix    
        excited_DM = qt.Qobj(excited)*qt.Qobj(excited).dag() # density matrix   
        
        self.states = qt.tensor(ground_DM,qt.Qobj([[1,0],[0,0]])) +qt.tensor(excited_DM,qt.Qobj([[0,0],[0,1]])) 

    # assume Hamiltonian only affects the even n ground states and odd n excited states
    def init_states_ground_fast(self):
        pops = np.zeros(self.N_bins)
        for i in range(0,self.N_bins,2):
            if i % 2 == 0:
                pops[i] = np.sqrt(self.initial_MB)[i]
            else: 
                #print(self.notch_function)
                pops[i] = 0
        
        pops = qt.Qobj(pops)
        self.states = pops*pops.dag() # density matrix   
        


    def save_states(self):
        time = 0
        self.checkpoints = [time]
        for laser in self.laserDict:
            time = laser[1].endTime
            self.checkpoints.append(time)

        for time in self.checkpoints:
            idx = int(time/self.dt)
            self.saved_states.append( self.result.states[idx])  
    
    """
    def evolve(self):
        opts = qt.Options(store_states=True)
        result = qt.mesolve(self.H, self.states,self.tlist, options = opts,progress_bar=True)
        self.result = result
        self.save_states()
    """
    def init_expect_oper(self,dims):
        if dims == 2:
            self.e_ops_1S = [qt.tensor(self.kets[n]*self.kets[n].dag(),qt.Qobj([[1,0],[0,0]])) for n in range(self.N_bins)] # ground 
            self.e_ops_2P = [qt.tensor(self.kets[n]*self.kets[n].dag(),qt.Qobj([[0,0],[0,1]])) for n in range(self.N_bins)] # excited
            self.e_ops = self.e_ops_1S + self.e_ops_2P
        elif dims == 3:
            self.e_ops_1S = [qt.tensor(self.kets[n]*self.kets[n].dag(),qt.Qobj([[1,0,0],[0,0,0],[0,0,0]])) for n in range(self.N_bins)] # ground 
            self.e_ops_2P = [qt.tensor(self.kets[n]*self.kets[n].dag(),qt.Qobj([[0,0,0],[0,1,0],[0,0,0]])) for n in range(self.N_bins)] # excited
            self.e_ops_PI = [qt.tensor(self.kets[n]*self.kets[n].dag(),qt.Qobj([[0,0,0],[0,0,0],[0,0,1]])) for n in range(self.N_bins)] # photo-ionised states
            self.e_ops = self.e_ops_1S + self.e_ops_2P + self.e_ops_PI

    """
    Set jump operator for photo-ionisation
    """
    def set_jump_operator_photoionisation(self,args):
        jump_coupling_strength = np.exp(-1j*(omega0-0.5*args["chirp"]*args["tlist"])*args["tlist"])*np.sqrt(photoionisation_cross_section*eps0*c/(2*omega0)*args["rabi"]**2*hbar_eV/Debye**2) # sqrt of the photoionisation rate
        self.c_ops = [[qt.tensor(ket*ket.dag(),self.qobj_dis),jump_coupling_strength] for ket in self.kets]

    """
    Set jump operator for annihilation
    """
    def set_jump_operator_annihilation(self,args):
        #rate = 1/142e3 # ps
        rate = 1/3 # ps
        jump_coupling_strength = np.sqrt(rate)
        qobj_dis = qt.Qobj([[0,0,0],[0,0,0],[1,0,0]])
        self.c_ops = [jump_coupling_strength*qt.tensor(ket*ket.dag(),qobj_dis) for ket in self.kets]
    
    
    """
    Set jump operator for spontaneous emission
    """
    def set_jump_operator_spontaneous_emission(self,args):
        #rate = 1/142e3 # ps
        rate = 1/3 # ps
        jump_coupling_strength = np.sqrt(rate)
        qobj_dis = qt.Qobj([[0,1,0],[0,0,0],[0,0,0]])
        self.c_ops = [jump_coupling_strength*qt.tensor(ket*ket.dag(),qobj_dis) for ket in self.kets]


    def evolve(self,dissipation=None):
        opts = qt.Options(store_states=True)
        self.saved_states.append(self.states)
        for i in range(len(self.laserDict)):
            laser = self.laserDict[i][1]
            args = {"chirp":np.asarray(laser.chirp(laser.tlist,None)),
                    "wavevector":np.asarray(laser.wavevector(laser.tlist,None)),
                    "rabi":np.asarray(laser.rabi(laser.tlist,None)),
                    "beating":np.asarray(laser.rabi_beating2(laser.tlist,None)),
                    "selector1":laser.selector1,
                    "selector2":laser.selector2,
                    "tlist":laser.tlist}
            #self.set_Hamiltonian_notched_MT4(args)
            #print(args["chirp"])

            #self.init_collapse_oper(laser)
            self.set_Hamiltonian_MT(args)
            if dissipation == "photoionisation":
                self.set_jump_operator_photoionisation(args)
            elif dissipation == "s.e.":
                self.set_jump_operator_spontaneous_emission(args)
            elif dissipation == "annihilation":
                self.set_jump_operator_annihilation(args)
            
            self.H.append(hbar*self.tensor_dissipative*(-1e5))
            
            result = qt.mesolve(self.H, self.saved_states[i],laser.tlist, e_ops=self.e_ops,c_ops=self.c_ops, options = opts,progress_bar=True)

            # temporary solution for getting expectation values
            self.saved_expect.append(np.asarray(result.expect))

            # temporarily not usable to get the number of grounds and exciteds due to working with a 3D system
            self.saved_states.append(result.states[-1])

