# SU(2) x SU(2)

This python package offers efficient simulation and data analysis routines for the SU(2) x SU(2) Principal Chiral model. The key feature offered is the integration of Fourier Acceleration into the Hybrid Monte Carlo algorithm which leads to a significant reduction in the degree of critical slowing down.

Currently the simulation is only supported for a two dimensional cubic lattice.

<!--
## Example

from SU2xSU2.SU2xSU2 import SU2xSU2
from SU2xSU2.calibrate_paras import calibrate

# define lattice and integration parameters as well as model parameter beta
model_paras = {'L':96, 'a':1, 'ell':15, 'eps':1/15, 'beta':1}
# find number of integration steps and their size (under the constraint that their product is 1) to get an acceptance rate in the interval [0.6, 0.75]
paras_calibrated = calibrate(model_paras, accel=True)

# make a model with the calibrated parameters
model = SU2xSU2(**paras_calibrated)
# define the simulation parameters, what observables should be measures and where the chain is stored
sim_paras = {'M':3000, 'thin_freq':1, 'burnin_frac':0.05, 'accel':True, 'measurements':[model.ww_correlation_func], 'chain_paths':['corfunc_beta1']}
# run simulation
model.run_HMC(**sim_paras) 

## Documentation
Read the docs [here](insert URL)
-->

## Installation 
To install ``SU2xSU2`` using ``pip`` run:

```bash
pip install SU2xSU2
```

<!--
## Attribution

Please cite the following papers if you found this code useful in your research:
```bash
    @article{}
```
-->

## Licence

``SU2xSU2`` is free software made available under the MIT License. For details see the `LICENSE` file.

## To DO
- move get_avg_error, corlength and main analysis functions (including effective mass and cost function) into one file
    - add 3 loop beta function to mass over lambda plot
- make file for all plotting functions
- data storage
    - manual data paths relative to current directory
    - readme files to describe contents of saved files
    - manual chain state path
- improve tests
- plotting style file
- possibly change the location of functions used in the main analysis
- fill in requirements.txt
- create documentation
- generalise simulation and data analysis to d-dimensional cubic lattice 