# STA663_FinalProject_GCHMM

:sweat_smile: 

This repository contains the following files:

### ipython notebook files:
AEMBP_GCHMM.ipynb: Optimized Generalized Baum-Welch (GBW), also called approximiate EM Belief Propogation (AEMBP) algorithm for for Graph-Coupled HMMs [fastest]

AEMBP.ipynb: Unoptimized GBW (or AEMBP)


Gibbs_GCHMM.ipynb: Optimized Gibbs Sampler algorithm for Graph-Coupled HMMs [fastest]<br />
Gibbs_numpy.ipynb: Numpy Gibbs Sampler with separate functions<br />
Gibbs_cython.ipynb: cythonized Gibbs Sampler with separate functions<br />
Gibbs_jit.ipynb: JIT version of Gibbs Sampler with separate functions<br />
Gibbs_naive.ipynb: naive version of Gibbs Sampler with separate functions<br />


### Data sets:
G: Dynamic social network graph (contains only upper triangular matrix) of 84 students in 107 days 
dim: (84, 84, 107) 

X: True latent states of infection 

dim: (84, 108) 

Y: Observed states of the appearance of 6 symptoms per student per day 

dim: (84, 6, 107)    


### Code generated pictures:
true.png, missing02.png,...etc

## Reproduce Code
To replicate the algorithm, please clone this repository, and run AEMBP_GCHMM.ipynb and Gibbs_GCHMM.ipynb

## Reproduce Report
To replicate the report, please clone this repository, and run in Terminal
```
$ make
```
All output files are generated in folder obj/ including *.pdf, *.aux, *.bbl, *.log, *.out .
