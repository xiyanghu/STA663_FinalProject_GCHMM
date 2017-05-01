# STA663_FinalProject_GCHMM

:sweat_smile: 

Examples folder contains the following files:

### ipython notebook files:
Example_Optimized_AEMBP.ipynb: Optimized Generalized Baum-Welch (GBW), also called approximiate EM Belief Propogation (AEMBP) algorithm for for Graph-Coupled HMMs [fastest]<br />
AEMBP.ipynb: Unoptimized GBW (or AEMBP)<br />

Example_Optimized_Gibbs.ipynb: Optimized Gibbs Sampler algorithm for Graph-Coupled HMMs [fastest]<br />
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
### option1
To replicate the algorithms, please clone this repository, go to Folder Examples and run Example_Optimized_AEMBP.ipynb and Example_Optimized_Gibbs.ipynb<br/> 
### option2
To replicate the algorithms using pre-wrapped packages, please clone this repository, go to Folder Packages/Gibbs or /GBW,  and run example.ipynb<br/> 

## Reproduce Report
To replicate the report, please clone this repository, and under the working directory including latex source files, run in Terminal
```
$ make
```
All output files are generated in folder obj/ including *.pdf, *.aux, *.bbl, *.log, *.out .
