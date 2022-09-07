def evolve(velocity_bins,omega0,detuning,chirp,rabi0,pulse_duration):
    
    rabi = lambda t, args: rabi0 * np.exp(-4*np.log(2)*(t-tcentre)**2/pulse_duration**2)
    final_states = []
    for vel in velocity_bins:
        
        H0 = hbar*qt.Qobj([[0,0],[0,omega0*vel/c +detuning*(1+vel/c)]])
        H_chirp = hbar*qt.Qobj([[0,0],[0,chirp*(1+vel/c)]])
        H_transition = 0.5*hbar*qt.sigmax()
        H = [H0,[H_chirp,tlist-tlist_centre],[H_transition,rabi]]

        result = qt.mesolve(H, psi0,tlist,c_ops,e_ops=projs)
        current_state = qt.Qobj([[result.expect[0][-1]],[result.expect[1][-1]]])
        final_states.append(current_state)
    
    return final_states 

# Take initial states of atoms and change the population in each velocity class due to photon momentum transfer
# for a proof of concept, assume the laser is going in the negative direction
# For hte full method, might have to decide which direction to scan through the states depending on which direction the photon is coming from.
def update_states(states, transitions):
    temp_states = states
    new_states = np.zeros(N_bins)
    for i in range(1,N_bins-1):
        transition = np.abs(transitions[i][1][0][0])
        new_states[i] = temp_states[i]*(1-transition) + temp_states[i+1]*transition
    return new_states


# Evolve with an arbitrary initial state, and return the evolved states
def evolve_train(velocity_bins,omega0,detuning,chirp,rabi0,pulse_duration,previous_states):
    rabi = lambda t, args: rabi0 * np.exp(-4*np.log(2)*(t-tcentre)**2/pulse_duration**2)
    final_states = []
    size = velocity_bins.size
    
    for i in range(size):
        initial_state = previous_states[i]
        vel = velocity_bins[i]
        
        H0 = hbar*qt.Qobj([[0,0],[0,omega0*vel/c +detuning*(1+vel/c)]])
        H_chirp = hbar*qt.Qobj([[0,0],[0,chirp*(1+vel/c)]])
        H_transition = 0.5*hbar*qt.sigmax()
        H = [H0,[H_chirp,tlist-tlist_centre],[H_transition,rabi]]

        result = qt.mesolve(H, initial_state,tlist,c_ops,e_ops=projs)
        final_states.append(qt.Qobj([[result.expect[0][-1]],[result.expect[1][-1]]]))
    
    return final_states 