e = 1.6e-19 #coulomb
h = 6.63e-34 *1e12/ e #eV ps
hbar = 1
hbar_eV = 6.582e-16 * 1e12 #eV ps, hbar divided by eV
c = 3e8 * 1e-10 # cm/ps
k = 8.617e-5 # [eV K^-1] Boltzmann's constant

wavelength = 243e-7 #cm
energy_splitting =  h*c/wavelength
omega0 = energy_splitting / hbar / hbar_eV