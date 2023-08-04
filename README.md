# SU(2) x SU(2)

This python package offers efficient simulation and data analysis routines for the SU(2) x SU(2) Principal Chiral model. The key feature offered is the integration of Fourier Acceleration into the Hybrid Monte Carlo algorithm which leads to a significant reduction in the degree of critical slowing down.

Currently the simulation is only supported for a two dimensional cubic lattice.

<!--
## Example

from SU2xSU2.SU2xSU2 import SU2xSU2

# define model and lattice parameters 
model_paras = {'L':40, 'a':1, 'ell':5, 'eps':1/5, 'beta':0.6}
model = SU2xSU2(**model_paras)
# define simulation parameters and measurements
sim_paras = {'M':500, 'thin_freq':1, 'burnin_frac':0.5, 'accel':True, 'measurements':[model.ww_correlation_func], 'chain_paths':['corfunc_chain.npy']}
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
- create documentation
- add tests
- once package is published and can be installed
    - check if example in README works
    - check if example.py works
    - data storage: Check that paths are relative to the to current working directory
- plotting
    - get latex error when plotting within analysis.py
    - no apparent option to add errorbar format '.' in style sheet 
    - include mplstyle file in stylelib/ to be used globally. Currently, the file needs to be copied manually into the directory. Possible approaches:
        - https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/__init__.py
        using https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/style/core.py
        - https://matplotlib.org/stable/tutorials/introductory/customizing.html#distributing-styles
        - https://stackoverflow.com/a/52997575
        - https://stackoverflow.com/questions/35851201/how-can-i-share-matplotlib-style
- generalize simulation and data analysis to d-dimensional cubic lattice 