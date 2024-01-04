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

# import pytest 

def test_nearest_neighbors():
    '''
    Checks if the nearest neighbor mask does indeed return the nearest neighbors when applied to the field.
    The test is performed on a D=4 lattice with L=16 (the memory requirement grows as L**D)
    '''
    D, L, a = 4, 16, 1
    model = SU2xSU2(D, L, a, ell=1, eps=1, beta=1)
    mask = model.NN_mask

    lattice_shape = tuple(np.repeat(L,D))
    field_shape = lattice_shape+(4,)
    field = np.random.standard_normal(field_shape)
    NNs = field[mask] # each position of the lattice contains the nearest neighbor matrices, starting with the D positive directions along the lattice, 
    # followed by the D negative directions (describes ordering along the last axis of NNs)

    # NN of random site
    c = np.random.randint(0,L,size=D)
    # check NN along positive directions
    assert np.array_equal(field[(c[0]+1)%L,c[1],c[2],c[3]], NNs[c[0],c[1],c[2],c[3],0])
    assert np.array_equal(field[c[0],(c[1]+1)%L,c[2],c[3]], NNs[c[0],c[1],c[2],c[3],1])
    assert np.array_equal(field[c[0],c[1],(c[2]+1)%L,c[3]], NNs[c[0],c[1],c[2],c[3],2])
    assert np.array_equal(field[c[0],c[1],c[2],(c[3]+1)%L], NNs[c[0],c[1],c[2],c[3],3])
    # check NN along negative directions
    assert np.array_equal(field[(c[0]-1)%L,c[1],c[2],c[3]], NNs[c[0],c[1],c[2],c[3],4])
    assert np.array_equal(field[c[0],(c[1]-1)%L,c[2],c[3]], NNs[c[0],c[1],c[2],c[3],5])
    assert np.array_equal(field[c[0],c[1],(c[2]-1)%L,c[3]], NNs[c[0],c[1],c[2],c[3],6])
    assert np.array_equal(field[c[0],c[1],c[2],(c[3]-1)%L], NNs[c[0],c[1],c[2],c[3],7])
    

def test_leapfrog():
    '''
    Checks if the leapfrog implementation (for standard and accelerated HMC) is reversible.
    '''
    D = np.random.randint(2,5)
    L = np.random.randint(2,16)
    lattice_shape = tuple(np.repeat(L,D))
    ell = np.random.randint(1,25)
    eps = 1/ell # forcing a fixed unit trajectory length 

    model = SU2xSU2(D, L, a=1, ell=ell, eps=eps, beta=1)
    field = np.random.standard_normal(lattice_shape+(4,))
    field_start = SU2.renorm(field)
    mom_start = np.random.standard_normal(lattice_shape+(3,))

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
    D = 2
    L = 8 # L = np.random.randint(2,100)
    beta = np.random.uniform(0.5, 2)
    model_paras = {'D':D, 'L':L, 'a':1, 'ell':4, 'eps':1/4, 'beta':beta}

    ### standard HMC ###
    def KE_per_site(phi, pi):
        L = phi.shape[0]
        D = len(phi.shape[:-1])
        K = 1/2 * np.sum(pi**2)
        return K/L**D
    
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
    allowed_err = 0.05
    check = ( (expected_val*(1-allowed_err)) <= avg <= (expected_val*(1+allowed_err)) )
    assert(check)

    ### accelerated HMC ###
    # calibrate number of integration steps and their size and run simulation
    paras_calibrated = calibrate(model_paras, accel=True)
    model = SU2xSU2(**paras_calibrated)
    A = model.kernel_inv_F()

    def KE_per_site_FA(phi, pi):
        L = phi.shape[0]
        D = len(phi.shape)
        # find magnitude of FT of each component of momentum in Fourier space. Then sum over all 3 components
        ax = tuple(np.arange(D))
        pi_F_mag = np.sum( np.abs(np.fft.fftn(pi, axes=ax))**2, axis=-1 ) # (lattice shape) 
        T = 1/(2*L**D) * np.sum(pi_F_mag*A) # sum over momentum Fourier lattice
        return T/L**D
    
    sim_paras = {'M':2000, 'burnin_frac':0.2, 'accel':True, 
            'measurements':[KE_per_site_FA], 'ext_measurement_shape':[(),], 'chain_paths':['kinetic_energy_FA']}
    model.run_HMC(**sim_paras) 
    
    # store data in file and delete it once computation finished 
    data = np.load('kinetic_energy_FA.npy')
    avg, err = get_avg_error(data)
    os.remove('kinetic_energy_FA.npy')

    expected_val = 3*1/2 # should this be different since the inverse kernel introduces a non-unit mass?
    allowed_err = 0.05
    check = ( (expected_val*(1-allowed_err)) <= avg <= (expected_val*(1+allowed_err)) )
    # assert(check)