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
G: Dynamic social network graph (contains only upper triangular matrix) of 84 students in 107 days <br />
   dim: (84, 84, 107) <br/>
X: True latent states of infection <br/>
   dim: (84, 108) <br/>
Y: Observed states of the appearance of 6 symptoms per student per day <br/>
   dim: (84, 6, 107) <br/> 


### Code generated pictures:
true.png, missing02.png,...etc

## Reproduce Code
# option1
To replicate the algorithm, please clone this repository, go to Folder Examples and run Example_Optimized_AEMBP.ipynb and Example_Optimized_Gibbs.ipynb
# option2


## Reproduce Report
To replicate the report, please clone this repository, and run in Terminal
```
$ make
```
All output files are generated in folder obj/ including *.pdf, *.aux, *.bbl, *.log, *.out .
