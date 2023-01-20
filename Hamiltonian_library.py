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
    """
    Old Hamiltonian with no momentum transfer functionality
    """
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
        H.append(-hbar*args["wavevector"]*omega0*self.tensor_vel/c*self.tensor_e+hbar*(self.tensor_enum**2*omega_recoil*(self.tensor_g+self.tensor_e))) # kinetic energy
        H.append([hbar*self.tensor_e+hbar*args["wavevector"]*self.tensor_vel/c*self.tensor_e,args["chirp"]]) # chirp terms
        H.append([-hbar*(0.5*self.tensor_ge +0.5*self.tensor_eg),args["rabi"]*args["selector1"]]) # time-dependent coupling terms               
        H.append([-hbar*(0.5*self.tensor_ge2 +0.5*self.tensor_eg2),args["rabi"]*args["selector2"]]) # time-dependent coupling terms               
        
        # DONT DELETE IN CASE I NEED TO CHANGE BACK TO THIS.
        # Used before I realised I can't solve for 1 train of pulses. It has to make a new sovler object for each pulse instead...
        #H.append([-hbar*omega0*self.tensor_vel/c*self.tensor_e,args["wavevector"]]) # velocity term
        #H.append([hbar*self.tensor_vel/c*self.tensor_e,args["chirp"]*args["wavevector"]]) # velocity term

        self.H = H


    # Sum of two pulses with two different detunings
    def set_Hamiltonian_notched_MT(self):

        self.init_pulse_cycle()
        
        # if you have coupling between  states more than 1 bin from each other, should probably find a way to make this non-zero...
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited

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

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))) # kinetic energy
        H.append([hbar*tensor_e,self.chirp]) # chirp terms
        H.append([-hbar*omega0*tensor_vel/c*tensor_e,self.wavevector]) # velocity term
        H.append([hbar*tensor_vel/c*tensor_e,self.chirp*self.wavevector]) # velocity term
        
        H.append([-hbar*(tensor_ge +tensor_eg),self.rabi_beating*self.func1]) # time-dependent coupling terms               
        H.append([-hbar*(tensor_ge2 +tensor_eg2),self.rabi_beating*self.func2]) # time-dependent coupling terms               

        self.H = self.H + H

    # beating with arctan definition
    def set_Hamiltonian_notched_MT3(self):        
        omega_recoil = 0#0.5*hbar*hbar_eV*self.wavenumber_value**2/(self.m/c**2) # 1/ps   
        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited

        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))) # kinetic energy
        H.append([hbar*tensor_e,self.chirp]) # chirp terms
        H.append([-hbar*omega0*tensor_vel/c*tensor_e,self.wavevector]) # velocity term
        H.append([hbar*tensor_vel/c*tensor_e,self.chirp*self.wavevector]) # velocity term
        
        H.append([-hbar*(tensor_ge +tensor_eg),self.rabi_beating2*self.func1]) # time-dependent coupling terms               
        H.append([-hbar*(tensor_ge2 +tensor_eg2),self.rabi_beating2*self.func2]) # time-dependent coupling terms               

        self.H = H
    
    """
    blabla
    """
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


    """
    For use in Krotov optimization
    """
    def set_Hamiltonian_Optimization(self,guess_phase,guess_envelope,detuning):        
        omega_recoil = 0
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor
        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e))-hbar*(self.omega0-(self.omega0+detuning)*(1+tensor_vel/c))*tensor_e)
        H.append([hbar*(1+tensor_vel/c)*tensor_e,guess_phase]) # chirp terms
        H.append([-hbar*(0.5*tensor_ge +0.5*tensor_eg),guess_envelope])
        
        self.H = H



    # for each new dissipation channel do:
    # 1. Expand the internal state arrays
    # 2. Define the coupling strength between the environment and system, along with the jump operator, and add them onto the current list of collapse operators
    # 3. Add new expectation operator to the current list of e_ops
    def include_SE_simple(self,flag_SE_simple=False):
        if flag_SE_simple:
            # dissipation is between already-existing states. So no need to expand dimensions.
            oper_SE_simple = qt.projection(self.internal_dims,0,1)
            rate_SE_simple = 1/3 # ps
            coupling_strength_SE = np.sqrt(rate_SE_simple)
            self.add_new_liouville_operator(oper_SE_simple,coupling_strength_SE)
            #self.proj_SE_simple = qt.ket2dm(qt.basis(self.internal_dims,self.internal_dims-1))
            #self.e_ops += [qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),self.proj_SE_simple) for n in range(self.N_bins)]
            #self.idx_e_ops["simple s.e."] = np.arange(self.order*self.N_bins,(self.order+1)*self.N_bins)
            #self.order +=1
    
    def include_SE_distributive(self,flag_SE_distributive=False):
        if flag_SE_distributive:
            # dissipation is between already-existing states. So no need to expand dimensions.
            oper_SE = qt.projection(self.internal_dims,0,1)
            rate_SE = 1/3 # ps
            coupling_strength_SE = np.sqrt(rate_SE)
            oper_SE_distributive = [0.6*qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),oper_SE)
                                    + 0.2*qt.tensor(self.kets_vel[n+1]*self.kets_vel[n].dag(),oper_SE)
                                    + 0.2*qt.tensor(self.kets_vel[n]*self.kets_vel[n+1].dag(),oper_SE) for n in range(self.N_bins-1)]
            self.c_ops += oper_SE_distributive
            #self.add_new_liouville_operator(oper_SE,coupling_strength_SE)
            #self.proj_SE = qt.ket2dm(qt.basis(self.internal_dims,self.internal_dims-1))
            #self.e_ops += [qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),self.proj_SE) for n in range(self.N_bins)]
            #self.idx_e_ops["s.e."] = np.arange(self.order*self.N_bins,(self.order+1)*self.N_bins)
            #self.order +=1


    def include_photoionisation(self, flag_photoionisation=False):
        if flag_photoionisation:
            self.internal_dims +=1 
            #self.add_dimension_to_system()
            oper_photoionisation = qt.projection(self.internal_dims,self.internal_dims-1,1)
            rate_photoionisation = 1/3 # ps
            coupling_strength_photoionisation = np.sqrt(rate_photoionisation)
            self.add_new_liouville_operator(oper_photoionisation,coupling_strength_photoionisation)
            self.proj_photoionisation = qt.ket2dm(qt.basis(self.internal_dims,self.internal_dims-1))
            self.e_ops +=   [qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),self.proj_photoionisation) for n in range(self.N_bins)]
            self.idx_e_ops["photoionisation"] = np.arange(self.order*self.N_bins,(self.order+1)*self.N_bins)

            self.order +=1


    def include_annihilation(self,flag_annihilation=False):
        if flag_annihilation:
            self.internal_dims +=1
            oper_annhilation = qt.projection(self.internal_dims,self.internal_dims-1,0)
            rate_annihilation = 1/3 # ps
            coupling_strength_annihilation = np.sqrt(rate_annihilation)
            self.add_new_liouville_operator(oper_annhilation,coupling_strength_annihilation)
            self.proj_annihilation = qt.ket2dm(qt.basis(self.internal_dims,self.internal_dims-1))
            self.e_ops += [qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),self.proj_annihilation) for n in range(self.N_bins)]
            self.idx_e_ops["ann."] = np.arange(self.order*self.N_bins,(self.order+1)*self.N_bins)            
            self.order +=1

 
    """
    add a new liouville operator to the list of current ones, given a liouville operator and corresponding coupling strength.
    The coupling strength can be a single value, or a function over time given as a list
    """
    def add_new_liouville_operator(self,jump_operator,coupling_strength,):
        if coupling_strength == type(list):
            new_op = [[qt.tensor(ket*ket.dag(),self.qobj_dis),coupling_strength] for ket in self.kets_vel]
            self.c_ops += new_op
        else:
            try:
                float(coupling_strength)
                new_op = [coupling_strength*qt.tensor(ket*ket.dag(),jump_operator) for ket in self.kets_vel]
                self.c_ops += new_op
            except ValueError:
                print("Environment coupling strength cannot be typecast to float") 


    def create_velocity_space(self):
        self.qobj_vel = qt.Qobj(np.diag(self.velocity_bins))
        self.qobj_n_prev = qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0))
        self.qobj_n_next = qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0))


    def create_internal_state_space(self):
        self.proj_1S = qt.ket2dm(qt.basis(self.internal_dims,0))
        self.e_ops +=  [qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),self.proj_1S) for n in range(self.N_bins)]
        self.idx_e_ops["1S"] = np.arange(self.order*self.N_bins,(self.order+1)*self.N_bins)
        self.order +=1
        
        self.proj_2P = qt.ket2dm(qt.basis(self.internal_dims,1))
        self.e_ops +=  [qt.tensor(self.kets_vel[n]*self.kets_vel[n].dag(),self.proj_2P) for n in range(self.N_bins)]
        self.idx_e_ops["2P"] = np.arange(self.order*self.N_bins,(self.order+1)*self.N_bins)
        self.order +=1
        
        self.proj_1S_to_2P = qt.projection(self.internal_dims,1,0)
        self.proj_2P_to_1S = qt.projection(self.internal_dims,0,1)


    def create_composite(self):
        self.create_velocity_space()

        self.include_annihilation(self.flag_annihilation)
        self.include_SE_simple(self.flag_SE_simple)
        self.include_SE_distributive(self.flag_SE_distributive)
        self.include_photoionisation(self.flag_photoionisation)
        
        self.create_internal_state_space()
        
        self.tensor_enum = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(self.internal_dims)) # enumerated tensor
        self.tensor_vel = qt.tensor(self.qobj_vel,qt.qeye(self.internal_dims))  
        
        self.tensor_g = qt.tensor(qt.qeye(self.N_bins),self.proj_1S) # ground 
        self.tensor_e = qt.tensor(qt.qeye(self.N_bins),self.proj_2P) # excited
        self.tensor_eg = qt.tensor(self.qobj_n_prev,self.proj_1S_to_2P) 
        self.tensor_ge = qt.tensor(self.qobj_n_next,self.proj_2P_to_1S) 
        self.tensor_eg2 = qt.tensor(self.qobj_n_prev,self.proj_2P_to_1S)
        self.tensor_ge2 = qt.tensor(self.qobj_n_next,self.proj_1S_to_2P)
    
    def evolve(self):
        pass



"""
NOT USED. LEGACY CODE


# UNUSED, originally meant to add a single extra state for photoionisation, but this is not compatible with QuTiP's Hamiltonian formulation
    def addDissipation(self,arr):
        new_arr = np.insert(arr,self.N_bins,np.zeros(self.N_bins),0)
        new_arr = np.insert(new_arr,self.N_bins,0,1)
        return new_arr

    # NOT USED
    def set_Hamiltonian_MT3(self):        
        tensor_vel = qt.tensor(qt.Qobj(np.diag(self.velocity_bins)),qt.qeye(2))     
        tensor_num = qt.tensor(qt.num(self.N_bins,offset=-self.N_bins//2+1),qt.qeye(2)) # enumerated tensor

        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n-1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n-1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        tensor_eg2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge2 = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(1,self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited

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
        tensor_g = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[1,0],[0,0]])) # ground 
        tensor_e = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n].dag() for n in range(self.N_bins)]),axis=0)),qt.Qobj([[0,0],[0,1]])) # excited
        tensor_eg = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n]*self.kets_vel[n+1].dag() for n in range(self.N_bins-1)]),axis=0)),qt.Qobj([[0,0],[1,0]])) # excited to ground
        tensor_ge = qt.tensor(qt.Qobj(np.sum(np.asarray([self.kets_vel[n+1]*self.kets_vel[n].dag() for n in range(self.N_bins-1)]),axis=0)),qt.Qobj([[0,1],[0,0]])) # ground to excited
        
        
        H = []
        H.append(hbar*(tensor_num**2*omega_recoil*(tensor_g+tensor_e) - self.omega0*tensor_vel/c*tensor_e)) # time-independent
        H.append([hbar*(1+tensor_vel/c)*laser.chirp0,laser.tlist-laser.tlist_centre]) # time-dependent terms
        H.append([hbar*(-0.5*tensor_eg-0.5*tensor_ge),laser.rabi]) # time-dependent coupling terms                

        opts = qt.Options(store_states=True)
        result = qt.mesolve(H, self.states,laser.tlist, options = opts)
        self.states = result.states[-1]


   def add_dimension_to_system(self):
        self.arr_1S = self.add_dimension(self.arr_1S)
        self.arr_2P = self.add_dimension(self.arr_2P)
        self.arr_1S_to_2P = self.add_dimension(self.arr_1S_to_2P)
        self.arr_2P_to_1S = self.add_dimension(self.arr_2P_to_1S)
        self.internal_dims += 1


    def add_dimension(self,arr):
        arr = np.insert(arr,self.internal_dims,0,axis=1)
        arr = np.append(arr,[0]*self.internal_dims,0)
        return arr
    
"""