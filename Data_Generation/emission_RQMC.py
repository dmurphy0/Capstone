import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import petitRADTRANS
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.planet import Planet
from petitRADTRANS.spectral_model import SpectralModel
from datetime import datetime
from tqdm import tqdm
import h5py
import astropy.constants as const   
import gc

from scipy.stats import qmc

startTime = datetime.now()


# Define sampler
sampler = qmc.Sobol(d=9, scramble=True, seed=42)     #this generates Sobol Sequences in 9 dimensions to sample data. 

# Generate n samples, use base 2 for maximum effectiveness
raw_samples = sampler.random(n=16384)  #16384 = 2^14

#[Temp_eq0, Temp_int1, Gamma(log)2, K_ir(log)3, g_ref(log)(cgs)4, CO(log)5, Met(log)6, R_p/R_s7, a/R_s(log)8]    #list of variables 

l_bounds = [500,  100, -4.6, -9.2,  5.3, -2.3, -1.3,  0.05,  0.7]
u_bounds = [3500, 500,  2.3,   0,   9.9,  4.6,  1.3,  0.25,  3.9] 
X_rqmc = qmc.scale(raw_samples, l_bounds, u_bounds)

#with h5py.File('emission_computation_set.h5', "r") as f:
 #   X_rqmc=f['emission_train_set'][:]    #load previous work to break into seperate chunks
   
planet = Planet.get('WASP-121 b')  #calls planet data for WASP-121 b for additional dataset 
#WASP 121-B parameter for testing
X_rqmc = [2409, 450, -1, -4.5, np.log(planet.reference_gravity), np.log(0.3), np.log(0.5), 0.12255, np.log(3.7844)]



n= 4096    #used 4 blocks of 4096 at a time ~8hours each

y_train_flux = []

pbar = tqdm(total=n)    #use progress bar

for i in range(1):   #0 to 4095, change i when splitting computation into parts. 

    inputs = X_rqmc[i,:]
    #inputs = X_rqmc    #toggle for individual planet data
    print(inputs)


    #can call these 3 parameters outside spectral_model mode: 
    rad_s = 10* cst.r_sun  #stellar radius (arbitrary initial)   
    rad_p = inputs[7]*rad_s  #planet radius (scaled)
    orbit_smaj = np.exp(inputs[8])*rad_s #semi major axis (scaled). (approximate roughly circular)

    spectral_model = SpectralModel(
        # Radtrans parameters
        pressures=np.logspace(-6, 2, 100),
        line_species=[   #equilibrium chemistry setting will calculate fractions and density profiles
            'H2O',
            'CO-NatAbund',
            'CH4',
            'CO2',
            'Na',
            'K'
        ],
        rayleigh_species=['H2', 'He'],
        gas_continuum_contributors=['H2--H2', 'H2--He'],
        wavelength_boundaries=[0.3, 5],

        scattering_in_emission=True, #accounts for scattering in emission spectra
    
        planet_radius=rad_p,   #[7] 
        reference_gravity=np.exp(inputs[4]), #[4] log g_ref 

        reference_pressure=1e-2, 

        # Star, system, orbit
        is_observed=False,  # return the flux observed at system_distance
        #system_distance= 10* cst.s_cst.light_year * 1e2,  # m to cm, used to scale the spectrum
     
        is_around_star=False,  # if True, calculate a PHOENIX stellar spectrum and add it to the emission spectrum
        #is_around_star = True
        #star_effective_temperature=5500,  # used to get the PHOENIX stellar spectrum model
        #star_radius= rad_s,  # arbitrary, as long as ratio link to "a" & "R_planet"
        
        orbit_semi_major_axis= orbit_smaj  ,  # 8

        temperature_profile_mode='guillot',   #use guillot temperature pressure profile
        temperature=inputs[0],   #this is equilibrium temp. Radtrans calculates T_irr from T_eq, T_int, Gamma and Kappa. 
        intrinsic_temperature=inputs[1],  
        guillot_temperature_profile_gamma=np.exp(inputs[2]),
        guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=np.exp(inputs[3]),

        # Mass fractions
        use_equilibrium_chemistry=True,
        filling_species={
            'H2': 37.0,
            'He': 12.0,

        },
        metallicity=np.exp(inputs[6]),  # times solar
        co_ratio=np.exp(inputs[5]),
    )

    wavelengths, flux = spectral_model.calculate_spectrum(
            mode='emission'
        )
    print(np.shape(flux),np.shape(wavelengths))
    plt.plot(wavelengths[0,:]*1e4, flux[0,:])
    
    plt.show()
    y_train_flux.append(flux)
    
    pbar.update(1)
    print('Duration: {}'.format(datetime.now() - startTime))    
    
    if i % 100 == 0:   #crude way of fixing any memory issues. It works though 
        gc.collect()
        print("collected")

#with h5py.File('temp_emission_x.h5', 'w') as f:             #comment back in to save datasets as h5 files
#    f.create_dataset('real_emission_input', data=X_rqmc[12288:16384])
#    f.create_dataset('real_emission_flux', data=y_train_flux)
