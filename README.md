![logo](https://github.com/JulianWack/SU2xSU2/raw/master/logo.png)

[![GitHub](https://img.shields.io/badge/GitHub-JulianWack%2FSU2xSU2-blue?logo=GitHub)](https://github.com/JulianWack/SU2xSU2)
[![DOI](https://zenodo.org/badge/668764614.svg)](https://zenodo.org/badge/latestdoi/668764614)
[![ArXiv](https://img.shields.io/badge/arXiv-2308.14628-%23B31B1B?logo=arxiv)](https://arxiv.org/abs/2308.14628)
[![Documentation Status](https://readthedocs.org/projects/su2xsu2/badge/?version=stable)](https://su2xsu2.readthedocs.io/en/stable/?badge=stable)
[![PyPI version](https://badge.fury.io/py/SU2xSU2.svg)](https://badge.fury.io/py/SU2xSU2)
[![Downloads](https://static.pepy.tech/badge/SU2xSU2)](https://pepy.tech/project/SU2xSU2)

This python package offers efficient simulation and data analysis routines for the $SU(2) \times SU(2)$ Principal Chiral model. The key feature offered is the integration of Fourier Acceleration into the Hybrid Monte Carlo algorithm which leads to a significant reduction in the degree of critical slowing down.

Currently the simulation is only supported for a two dimensional cubic lattice.

## Installation 
To install ``SU2xSU2`` using ``pip`` run:

```bash
pip install SU2xSU2
```
Its is recommended to work in a virtual environment. The package comes with a custom style sheet which is used by default.


## Documentation
Read the docs [here](https://su2xsu2.readthedocs.io/).


## Example
A basic example showing how to set up a simulation using Fourier accelerated HMC to measure the wall-to-wall correlation function.
Further examples can be found [here](https://su2xsu2.readthedocs.io/en/stable/usage.html#examples).
```python
from SU2xSU2.SU2xSU2 import SU2xSU2

# define model and lattice parameters 
model_paras = {'D':2, L':40, 'a':1, 'ell':5, 'eps':1/5, 'beta':0.6}
model = SU2xSU2(**model_paras)
# define simulation parameters and measurements
sim_paras = {'M':500, 'burnin_frac':0.5, 'accel':True, 'measurements':[model.ww_correlation_func], 'chain_paths':['corfunc_chain.npy']}
model.run_HMC(**sim_paras) 
```


## Attribution

Please cite the following paper if you found this code useful in your research:
```bash
@misc{wack2023FAHMC,
    title={Hybrid Monte Carlo Simulation with Fourier Acceleration of the $N=2$ Principal Chiral Model in two Dimensions}, 
    author={Horsley, Roger and Pendleton, Brian and Wack, Julian},
    year={2023},
    eprint={2308.14628},
    archivePrefix={arXiv},
    primaryClass={hep-lat}
}
```


## Licence

``SU2xSU2`` is free software made available under the MIT License. For details see the `LICENSE` file.

## To Do
- Runtime warning in correlations l.64
- Check if coupling expansions and scaling predictions depend on D
- generalize to D dimensions when using acceleration 
<!-- check self.L**2 -> self.L**self.D (test equpartition, ) and FFT from 2 to D dimensional (also in test_simulation.test_equipartition() )-->