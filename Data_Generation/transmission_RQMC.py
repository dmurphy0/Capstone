#1D ML Spectrum Emulator for Exoplanets
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

from scipy.stats import qmc

startTime = datetime.now()



# Define your 5D sampler
sampler = qmc.Sobol(d=5, scramble=True, seed=42)

# Generate exactly 2048 for optimal Sobol properties
raw_samples = sampler.random(n=2048)

# Scale to your physical bounds
# bounds = [Temp, Radius, Gravity, CO, Metallicity]
R_jup = (const.R_jup.value)*100

l_bounds = [500, 0.7*R_jup, 200, 0.1, 0.1]
u_bounds = [3500, 2.5*R_jup, 20000, 100, 1.2]
X_rqmc = qmc.scale(raw_samples, l_bounds, u_bounds)



n=2048
y_train_radii = []


wl_timeseries=np.linspace((float(0.3*(10**(-6))/3912)),float(15*(10**(-6))/3912),3912)

pbar = tqdm(total=n)

for i in range(2048): #7seconds per iteration approx, looking at 3hrs for 1000
    
    inputs = X_rqmc[i,:]
    
    print(inputs)

    spectral_model = SpectralModel(
        
        # Radtrans parameters
        pressures=np.logspace(-6, 2, 100),
        line_species=[
            'H2O',
            'CO-NatAbund',
            'CH4',
            'CO2',
            'Na',
            'K' #add more spdata.h5ecies later
        ],
        rayleigh_species=['H2', 'He'],
        gas_continuum_contributors=['H2--H2', 'H2--He'],
        wavelength_boundaries=[0.3, 15],
    
        temperature=inputs[0],  
        planet_radius=inputs[1],
        reference_gravity=inputs[2],
        metallicity=inputs[3],  # [M/H]
        co_ratio=inputs[4],  # C/O ratio
        reference_pressure=0.01,     #assume isothermal pressure

        # Mass fractions
        imposed_mass_fractions={  # these can also be arrays of the same size as pressures
            'H2O': 1e-3,
            'CO-NatAbund': 1e-2,
            'CH4': 1e-5,
            'CO2': 1e-4,
            'Na': 1e-4,
            'K': 1e-6
        },
        filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
            'H2': 37,
            'He': 12
        }
    )

    wavelengths, transit_radii = spectral_model.calculate_spectrum(
        mode='transmission'
    )

    y_train_radii.append(transit_radii)

    pbar.update(1)

    print('Duration: {}'.format(datetime.now() - startTime))
    
pbar.close()

#with h5py.File('RQMC_transmission_accident.h5', 'w') as f:
#    f.create_dataset('RQMC_transmission_input', data=X_rqmc)
#    f.create_dataset('RQMC_transmission_radii', data=y_train_radii)
