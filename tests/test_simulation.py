# To ignore deprecation warnings, run
# pytest -W ignore::PendingDeprecationWarning
# Otherwise just
# pytest

import os
import numpy as np

from SU2xSU2.SU2xSU2 import SU2xSU2
import SU2xSU2.SU2_mat_routines as SU2
from SU2xSU2.calibrate_paras import calibrate
from SU2xSU2.analysis import get_avg_error

import pytest 

def test_nearest_neighbors():
    '''
    Checks if the nearest neighbor mask works does indeed return the nearest neighbors when applied to the lattice.
    '''
    L, a = 16, 1
    model = SU2xSU2(L, a, ell=1, eps=1, beta=1)
    mask = model.NN_mask
    field = np.random.standard_normal((L,L,4))
    NNs = field[mask] # each position of the lattice contains the nearest neighbor matrices in the order right, left, top, bottom on axis 2

    # NN of random site [i,j]
    i, j = np.random.randint(0,L), np.random.randint(0,L)
    assert np.array_equal(field[i,(j+1)%L],NNs[i,j,0]) # right
    assert np.array_equal(field[i,(j-1)%L],NNs[i,j,1]) # left
    assert np.array_equal(field[(i-1)%L,j],NNs[i,j,2]) # top
    assert np.array_equal(field[(i+1)%L,j],NNs[i,j,3]) # bottom


def test_leapfrog():
    '''
    Checks if the leapfrog implementation (for standard and accelerated HMC) is reversible.
    '''
    L = np.random.randint(2,100)
    ell = np.random.randint(1,25)
    eps = 1/ell # forcing a fixed unit trajectory length 

    model = SU2xSU2(L, a=1, ell=ell, eps=eps, beta=1)
    field = np.random.standard_normal((L,L,4))
    field_start = SU2.renorm(field)
    mom_start = np.random.standard_normal((L,L,3))

    # standard HMC
    field_end, mom_end = model.leapfrog(field_start, mom_start)
    field_start_rev, mom_start_rev = model.leapfrog(field_end, -mom_end)
    # expect field_start_rev = field_start and mom_start_rev = -mom_start
    assert np.allclose(field_start.flatten(), field_start_rev.flatten())
    assert np.allclose(mom_start.flatten(), -mom_start_rev.flatten())

    # Fourier Acceleration HMC
    field_end, mom_end = model.leapfrog_FA(field_start, mom_start)
    field_start_rev, mom_start_rev = model.leapfrog_FA(field_end, -mom_end)
    # expect field_start_rev = field_start and mom_start_rev = -mom_start
    assert np.allclose(field_start.flatten(), field_start_rev.flatten())
    assert np.allclose(mom_start.flatten(), -mom_start_rev.flatten())


def test_equipartition():
    '''
    Checks if the equipartition theorem (for standard and accelerated dynamics) is satisfied. 
    An SU2 matrix has 3 DOF such that with k_b T = 1, the average kinetic energy per site is 3 * 1/2.
    The test is considered to be passed if the computed average kinetic energy lies within +/-5% of the expected value.
    When not performing enough measurements or not rejecting enough burn in, it is possible that the test fails.
    To assure that the test runs quickly, a small lattice size is used.
    '''    
    L = 16 # L = np.random.randint(2,100)
    beta = np.random.uniform(0.5, 2)
    model_paras = {'L':L, 'a':1, 'ell':4, 'eps':1/4, 'beta':beta}

    ### standard HMC ###
    def KE_per_site(phi, pi):
        L = phi.shape[0]
        K = 1/2 * np.sum(pi**2)
        return K/L**2
    
    # calibrate number of integration steps and their size and run simulation
    paras_calibrated = calibrate(model_paras, accel=False)
    model = SU2xSU2(**paras_calibrated)
    sim_paras = {'M':2000, 'burnin_frac':0.2, 'accel':False, 
            'measurements':[KE_per_site], 'ext_measurement_shape':[(),], 'chain_paths':['kinetic_energy']}
    model.run_HMC(**sim_paras) 
    
    # store data in file and delete it once computation finished 
    data = np.load('kinetic_energy.npy')
    avg, err = get_avg_error(data)
    os.remove('kinetic_energy.npy')

    expected_val = 3*1/2
    allowed_deviation = expected_val * 0.05
    check = ( (expected_val-allowed_deviation) <= avg <= (expected_val+allowed_deviation))
    assert(check)

    ### accelerated HMC ###
    # calibrate number of integration steps and their size and run simulation
    paras_calibrated = calibrate(model_paras, accel=True)
    model = SU2xSU2(**paras_calibrated)
    A = model.kernel_inv_F()
    def KE_per_site_FA(phi, pi):
        L = phi.shape[0]
        # find magnitude of FT of each component of momentum in Fourier space. Then sum over all 3 components
        pi_F_mag = np.sum( np.abs(np.fft.fft2(pi, axes=(0,1)))**2, axis=-1 ) # (L,L) 
        T = 1/(2*L**2) * np.sum(pi_F_mag*A) # sum over momentum Fourier lattice
        return T/L**2
    sim_paras = {'M':2000, 'burnin_frac':0.2, 'accel':True, 
            'measurements':[KE_per_site_FA], 'ext_measurement_shape':[(),], 'chain_paths':['kinetic_energy_FA']}
    model.run_HMC(**sim_paras) 
    
    # store data in file and delete it once computation finished 
    data = np.load('kinetic_energy_FA.npy')
    avg, err = get_avg_error(data)
    os.remove('kinetic_energy_FA.npy')

    check = ( (expected_val-allowed_deviation) <= avg <= (expected_val+allowed_deviation))
    assert(check)