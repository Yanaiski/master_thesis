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

class HamiltonianClass():
    def  set_Hamiltonian_noCMT(self,laserLabel):
        laser = self.laserDict[laserLabel]

        tensor1 = qt.tensor(qt.qeye(self.N_bins),qt.Qobj([[0,0],[0,1]]))
        tensorn = qt.tensor(qt.num(self.N_bins,offset=1),qt.Qobj([[0,0],[0,1]]))
        tensord = qt.tensor(qt.qeye(self.N_bins),qt.sigmax())
        vel_squared = qt.Qobj(self.velocity_bins)*qt.Qobj(self.velocity_bins).dag()
        

        H0 = hbar*(laser.detuning+laser.direction*self.velocity_bins[0]/c*(laser.omega0+laser.detuning))*tensor1
        Hv0 = hbar * self.dv * (laser.omega0 + laser.detuning) * laser.direction/c * tensorn
        H_v_chirp = hbar*laser.chirp*laser.direction*self.dv/c*tensorn
        H_chirp = hbar * laser.chirp*(1+laser.direction*self.velocity_bins[0]/c)*tensor1
        H_transition = 0.5*hbar*tensord

        H = [H0,Hv0,[H_chirp+H_v_chirp,laser.tlist-laser.tlist_centre],[H_transition,laser.rabi]]
        
        opts = qt.Options(store_states=True)
        result = qt.mesolve(H, self.states,laser.tlist, options = opts)
        self.states = result.states[-1]
            
    # Momentum Transfer included
    # i.e velocity bins are no longer independent from each other
    def set_Hamiltonian_MT(self,args):   
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        H = []
        H.append(hbar*(self.tensor_enum**2*omega_recoil*(self.tensor_g+self.tensor_e))) # kinetic energy
        H.append([hbar*self.tensor_e,args["chirp"]]) # chirp terms
        H.append([-hbar*omega0*self.tensor_vel/c*self.tensor_e,args["wavevector"]]) # velocity term
        H.append([hbar*self.tensor_vel/c*self.tensor_e,args["chirp"]*args["wavevector"]]) # velocity term
        
        H.append([-hbar*(0.5*self.tensor_ge +0.5*self.tensor_eg),args["rabi"]*args["selector1"]]) # time-dependent coupling terms               
        H.append([-hbar*(0.5*self.tensor_ge2 +0.5*self.tensor_eg2),args["rabi"]*args["selector2"]]) # time-dependent coupling terms               
        
        self.H = H


    # Sum of two pulses with two different detunings
    def set_Hamiltonian_notched_MT(self):

        self.init_pulse_cycle()
        
        # if you have coupling between  states more than 1 bin from each other, should probably find a way to make this non-zero...
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited

        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))) # kinetic energy
        H.append([hbar*tensor_e,self.chirp]) # chirp terms
        H.append([-hbar*np.pi/2*tensor_e,self.notch_function])
        H.append([-hbar*omega0*tensor_vel/c*tensor_e,self.wavevector]) # velocity term
        H.append([hbar*tensor_vel/c*tensor_e,self.chirp*self.wavevector]) # velocity term
        
        H.append([-hbar*(0.5*tensor_ge +0.5*tensor_eg),self.rabi*self.func1]) # time-dependent coupling terms               
        H.append([-hbar*(0.5*tensor_ge2 +0.5*tensor_eg2),self.rabi*self.func2]) # time-dependent coupling terms               

        self.H = H
        
    # sin beating definition
    def set_Hamiltonian_notched_MT2(self):       
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))) # kinetic energy
        H.append([hbar*tensor_e,self.chirp]) # chirp terms
        H.append([-hbar*omega0*tensor_vel/c*tensor_e,self.wavevector]) # velocity term
        H.append([hbar*tensor_vel/c*tensor_e,self.chirp*self.wavevector]) # velocity term
        
        H.append([-hbar*(tensor_ge +tensor_eg),self.rabi_beating*self.func1]) # time-dependent coupling terms               
        H.append([-hbar*(tensor_ge2 +tensor_eg2),self.rabi_beating*self.func2]) # time-dependent coupling terms               

        self.H = H

    # beating with arctan definition
    def set_Hamiltonian_notched_MT3(self):        
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited

        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))) # kinetic energy
        H.append([hbar*tensor_e,self.chirp]) # chirp terms
        H.append([-hbar*omega0*tensor_vel/c*tensor_e,self.wavevector]) # velocity term
        H.append([hbar*tensor_vel/c*tensor_e,self.chirp*self.wavevector]) # velocity term
        
        H.append([-hbar*(tensor_ge +tensor_eg),self.rabi_beating2*self.func1]) # time-dependent coupling terms               
        H.append([-hbar*(tensor_ge2 +tensor_eg2),self.rabi_beating2*self.func2]) # time-dependent coupling terms               

        self.H = H
    
    def set_Hamiltonian_notched_MT4(self,args):        
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        H = []
        H.append(hbar*(self.tensor_num**2*omega_recoil*(self.tensor_g+self.tensor_e))) # kinetic energy
        H.append([hbar*self.tensor_e,args["chirp"]]) # chirp terms
        H.append([-hbar*omega0*self.tensor_vel/c*self.tensor_e,args["wavevector"]]) # velocity term
        H.append([hbar*self.tensor_vel/c*self.tensor_e,args["chirp"]*args["wavevector"]]) # velocity term
        
        H.append([-hbar*(self.tensor_ge +self.tensor_eg),args["beating"]*args["selector1"]]) # time-dependent coupling terms               
        H.append([-hbar*(self.tensor_ge2 +self.tensor_eg2),args["beating"]*args["selector2"]]) # time-dependent coupling terms               

        self.H = H

    def set_Hamiltonian_Optimization(self,guess_phase,guess_envelope,detuning):        
        omega_recoil = 0
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor
        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))-hbar*(self.omega0-(self.omega0+detuning)*(1+tensor_vel/c))*tensor_e)
        H.append([hbar*(1+tensor_vel/c)*tensor_e,guess_phase]) # chirp terms
        H.append([-hbar*(0.5*tensor_ge +0.5*tensor_eg),guess_envelope])
        
        self.H = H

    def set_Hamiltonian_MT_dissipation(self):   
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        H = []
        H.append(hbar*(self.tensor_num**2*omega_recoil*(self.tensor_g+self.tensor_e))) # kinetic energy
        H.append([hbar*self.tensor_e,self.chirp]) # chirp terms
        H.append([-hbar*omega0*self.tensor_vel/c*self.tensor_e,self.wavevector]) # velocity term
        H.append([hbar*self.tensor_vel/c*self.tensor_e,self.chirp*self.wavevector]) # velocity term
        
        H.append([-hbar*(0.5*self.tensor_ge +0.5*self.tensor_eg),self.rabi*self.func1]) # time-dependent coupling terms               
        H.append([-hbar*(0.5*self.tensor_ge2 +0.5*self.tensor_eg2),self.rabi*self.func2]) # time-dependent coupling terms               

        self.H = H

    def createTensors(self, isDissipative):
        vel_arr = np.diag(self.velocity_bins)
        nn_arr = np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)
        eg_arr = np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)
        ge_arr = np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)
        num_internal_states = 2 + int(isDissipative)
        # If dissipation is included (annhilation, photoionisation etc.), then insert an additional state for this state
        if isDissipative:
            qobj_g = qt.Qobj([[1,0,0],[0,0,0],[0,0,0]])
            qobj_e = qt.Qobj([[0,0,0],[0,1,0],[0,0,0]])
            qobj_ge = qt.Qobj([[0,0,0],[1,0,0],[0,0,0]])
            qobj_eg = qt.Qobj([[0,1,0],[0,0,0],[0,0,0]])
            self.qobj_dis = qt.Qobj([[0,0,0],[0,0,0],[0,1,0]]) #e/g to ionization
            
        else: 
            qobj_g = qt.Qobj([[1,0],[0,0]])
            qobj_e = qt.Qobj([[0,0],[0,1]])
            qobj_ge = qt.Qobj([[0,0],[1,0]])
            qobj_eg = qt.Qobj([[0,1],[0,0]])
            self.c_ops = []
        # I think I am forced to add a photoionisation state for each velocity state. Meaning each tensor is now a composite of a N_bins size array and a 3x3 matrix
        self.tensor_enum = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(num_internal_states)) # enumerated tensor
        self.tensor_vel = qt.tensor(qt.Qobj(vel_arr),qt.qeye(num_internal_states))  
        self.tensor_g = qt.tensor(qt.Qobj(nn_arr),qobj_g) # ground 
        self.tensor_e = qt.tensor(qt.Qobj(nn_arr),qobj_e) # excited
        self.tensor_eg = qt.tensor(qt.Qobj(eg_arr),qobj_ge) # excited to ground
        self.tensor_ge = qt.tensor(qt.Qobj(ge_arr),qobj_eg) # ground to excited
        self.tensor_eg2 = qt.tensor(qt.Qobj(eg_arr),qobj_eg) # excited to ground
        self.tensor_ge2 = qt.tensor(qt.Qobj(ge_arr),qobj_ge) # ground to excited

    # UNUSED, originally meant to add a single extra state for photoionisation, but this is not compatible with QuTiP's Hamiltonian formulation
    def addDissipation(self,arr):
        new_arr = np.insert(arr,self.N_bins,np.zeros(self.N_bins),0)
        new_arr = np.insert(new_arr,self.N_bins,0,1)
        return new_arr

    def evolve(self):
        pass



"""
NOT USED. LEGACY CODE


    # NOT USED
    def set_Hamiltonian_MT3(self):        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n-1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n-1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited

        states = self.states
        for laser in self.laserDict:
            chirp = np.asarray(laser[1].chirp(laser[1].tlist,None))
            wavevector = laser[1].unit_wavevector
            rabi = np.asarray(laser[1].rabi(laser[1].tlist,None))
            
            omega_recoil = 0#0.5*hbar*hbar_eV*laser[1].wavenumber_value**2/(self.m/c**2) # 1/ps
            H = []
            
            H.append(hbar*((tensor_num**2*omega_recoil*(tensor_g+tensor_e)) +(self.omega0-(laser[1].omega0+laser[1].detuning)*(1+tensor_vel/c*wavevector))*tensor_e)) # time-independent
            
            H.append([hbar*(1+tensor_vel/c*wavevector)*tensor_e,chirp]) # chirp terms
            if wavevector == 1:
                H.append([-hbar*(0.5*tensor_ge +0.5*tensor_eg),rabi]) # time-dependent coupling terms               
            else:
                H.append([-hbar*(0.5*tensor_ge2 +0.5*tensor_eg2),rabi]) # time-dependent coupling terms               
            
            opts = qt.Options(store_states=True,nsteps=10000)
            result = qt.mesolve(H, states,laser[1].tlist, options = opts,progress_bar=True)
            
            states = result.states[-1]
            self.saved_states.append(states)
        self.states = states

    # No longer used
    def set_Hamiltonian_MT(self,laserLabel):
        laser = self.laserDict[0][1]

        omega_recoil = 0.5*hbar*hbar_eV*laser.wavenumber_value**2/(self.m/c**2) # 1/ps   
        print(omega_recoil)
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))        
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=1),qt.qeye(2)) # enumerated tensor
        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n]*self.kets[n+1].dag() for n in range(self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets[n+1]*self.kets[n].dag() for n in range(self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e) - self.omega0*tensor_vel/c*tensor_e)) # time-independent
        H.append([hbar*(1+tensor_vel/c)*laser.chirp0,laser.tlist-laser.tlist_centre]) # time-dependent terms
        H.append([hbar*(-0.5*tensor_eg-0.5*tensor_ge),laser.rabi]) # time-dependent coupling terms                

        opts = qt.Options(store_states=True)
        result = qt.mesolve(H, self.states,laser.tlist, options = opts)
        self.states = result.states[-1]
"""