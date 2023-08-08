Usage
=====

.. _installation:

Installation
------------
To install ``SU2xSU2`` using ``pip`` run

.. code-block:: bash

    (.venv) $ pip install SU2xSU2

The dependencies of the package are detailed in ``requirements.txt``. To install these download the file and run

.. code-block:: bash

    pip install -r requirements.txt

Its is recommended to work in a virtual environment.

.. _examples:

Examples
--------
.. code-block:: python

    import numpy as np
    from SU2xSU2.SU2xSU2 import SU2xSU2
    from SU2xSU2.calibrate_paras import calibrate
    import analysis
    import plotting

    ### basic model creation and simulation call signature ###
    # measuring and plotting the correlation function is used in this example but the structure is identical for other observables  
    model_paras = {'L':40, 'a':1, 'ell':5, 'eps':1/5, 'beta':0.6} # define lattice and integration parameters as well as model parameter beta
    # find number of integration steps and their size (under the constraint that their product is 1) to get an acceptance rate in the interval [0.6, 0.75]
    paras_calibrated = calibrate(model_paras, accel=True)
    # make a model with the calibrated parameters
    model = SU2xSU2(**paras_calibrated)

    # define the simulation parameters, what observables should be measures and where the chain is stored
    sim_paras = {'M':500, 'thin_freq':1, 'burnin_frac':0.5, 'accel':True, 
                'measurements':[model.ww_correlation_func], 'chain_paths':['corfunc_test'],
                'chain_state_dir':'corfunc_test/chain_state/'}
    # run simulation
    model.run_HMC(**sim_paras) 

    # find ensemble average of the measurement chain and make plot
    avg, err = analysis.get_avg_error(np.load('corfunc_test.npy'))
    analysis.get_corlength(avg, err, 'corfunc_processed')
    plotting.correlation_func_plot('corfunc_processed.npy', 'plots/corfunc.pdf')

    # optionally can continue the previous chain
    sim_paras = {'M':500, 'thin_freq':1, 'burnin_frac':0.0, 'accel':True, 
                'measurements':[model.ww_correlation_func], 'chain_paths':['corfunc_test_continue'],
                'starting_config_path':'corfunc_test/chain_state/config.npy', 'RNG_state_path':'corfunc_test/chain_state/RNG_state.obj',
                'chain_state_dir':'corfunc_test/chain_state/'}
    model.run_HMC(**sim_paras) 


    ### compute internal energy density and plot it to compare it to coupling expansions ###
    betas = np.linspace(0.1, 4, 10)
    analysis.internal_energy_coupling_exp(betas, 16, 5000, 0.1, chaindata_pathbase='energy_data/', simdata_path='energy.txt', plot_path='energy_exp.pdf')


    ### mass over lambda ratio ###
    # value pairs which largely avoid finite size effects
    Ls = [40, 40, 64, 64, 64, 96, 96, 160, 160, 224, 400, 512, 700]
    betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.4])
    analysis.mass_lamdba(betas, Ls, 1e5, 0.02)


    ### acceleration mass grid search ###
    xi = 7.93
    masses = 1 / np.linspace(0.1*xi, 3*xi, num=10, endpoint=True)
    analysis.acceleration_mass_search(1e4, 0.05, 1, 96, xi, masses)


    ### critical slowing down ###
    # assuming the data produced in 'analysis.mass_lamdba' was stored at the default locations
    analysis.critical_slowingdown(1e5, 0.05)
