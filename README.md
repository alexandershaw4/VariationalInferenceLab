# VariationalInferenceLab

**Thermodynamic Variational Laplace and Bayesian Model Fitting Toolkit**

**Author:** Alexander D. Shaw
**Lab:** Computational Psychiatry & Neuropharmacological Systems Lab, University of Exeter
**Website:** https://cpnslab.com

---

## Overview

`aLogLikeFit` is a research MATLAB repository for fitting nonlinear generative models using **Variational Laplace**, **thermodynamic integration**, **generalised coordinates**, **hierarchical empirical Bayes**, and related Bayesian optimisation methods.

The repository is primarily designed for methodological development in:

* Dynamic Causal Modelling (DCM)
* Neural mass and thalamo-cortical modelling
* Computational psychiatry
* M/EEG spectral and time-series model fitting
* Hierarchical Bayesian inference and PEB-style analysis
* Active inference and expected free energy methods
* Neuro-inspired AI and adaptive control

This is not a polished toolbox with a single fixed API. It is a living research codebase containing core algorithms, experimental variants, demonstrations, and exploratory extensions.

---

## Core Idea

The central aim is to fit nonlinear models of the form:

```math
y = f(\theta, u, M) + \varepsilon
```

where:

* `y` is observed data
* `f` is a nonlinear forward or generative model
* `theta` are unknown parameters or latent states
* `u` are inputs or experimental conditions
* `M` is a model structure
* `epsilon` is observation noise

Most routines estimate posterior parameter distributions using a Laplace approximation:

```math
q(\theta) \approx \mathcal{N}(\mu, \Sigma)
```

with extensions for annealing, structured precision, low-rank noise, generalised coordinates, active inference, and hierarchical shrinkage.

---

## Main Features

### Thermodynamic Variational Laplace

Thermodynamic VL uses an annealing or temperature schedule to improve optimisation stability and reduce sensitivity to local minima. This is useful when fitting nonlinear dynamical systems with rugged posteriors or difficult likelihood surfaces.

Key files:

* `VL/fitVariationalLaplaceThermo.m`
* `VL/fitVariationalLaplaceThermo_GC.m`
* `other_experiments/fitVariationalLaplaceThermoStable.m`
* `other_experiments/fitVariationalLaplaceThermoStruct.m`

---

### Generalised Coordinates

Generalised-coordinate variants support inference over trajectories, derivatives, and local dynamics, making them relevant for DCM-style state-space models and dynamic systems.

Key files:

* `VL/fitVariationalLaplaceThermo_GC.m`
* `other_experiments/fitVariationalLaplaceThermo_GClam.m`
* `other_experiments/dcm_vl_gc.m`
* `other_experiments/dcm_vl_gc_time.m`

Relevant notes:

* `VL/VL_in_GeneralisedCoordinates.pdf`
* `VL/VL_in_GC_new.pdf`
* `VL/thermoVL_equations.pdf`

---

### Hierarchical VL and PEB / ARD

The repository contains several routines for hierarchical inference, empirical Bayes, and automatic relevance determination. These are useful for group-level modelling, shrinkage, parameter-field inference, and predictor selection.

Key files and folders:

* `other_experiments/fitHierarchicalVL.m`
* `other_experiments/Wrapper_AlogLikeDCM_fitHierarchicalVL.m`
* `PEB/peb_ard_with_stats_var.m`
* `PEB_ARD_general/`

The `PEB_ARD_general/` folder contains a more self-contained framework for PEB-style ARD analysis, including demonstration scripts, plotting routines, cross-validation, and prediction functions.

---

### Low-Rank and Structured Noise Models

Several routines explore alternatives to simple independent Gaussian noise, including low-rank covariance models, heteroscedastic precision updates, radial precision fields, and structured observation noise.

Key files:

* `other_experiments/fitVL_LowRankNoise.m`
* `other_experiments/fitVL_LowRankNoise2.m`
* `other_experiments/fitVariationalLaplaceThermoRadialPrecision.m`
* `other_experiments/update_hyper.m`
* `other_experiments/condFIM.m`

These are particularly relevant for high-dimensional observations such as spectra, image-like data, or dense time-frequency representations.

---

### DCM and Neural Mass Model Fitting

The repository includes wrappers and utilities for applying ThermoVL-style fitting to DCM and neural mass models.

Key files:

* `other_experiments/aFitDCM.m`
* `other_experiments/dcm_vl_gc.m`
* `other_experiments/dcm_vl_gc_time.m`
* `DIP/dcm_moga.m`
* `DIP/dip_hybrid_twocmp.m`
* `DIP/run_dip_as.m`

These routines are experimental and may assume local model structures or project-specific conventions.

---

### DEM-Style Inference

The `DEM/` folder contains an implementation of thermodynamic VL ideas in a Dynamic Expectation Maximisation-style setting.

Key files:

* `DEM/fitDEM_ThermoVL.m`
* `DEM/dem_test/`

---

### Bayesian SVM

The `SVM/` folder contains an experimental Bayesian support vector machine formulation using thermodynamic Bayesian fitting.

Key file:

* `SVM/fitThermoBayesSVM.m`

This is exploratory and intended as a proof-of-principle extension of the ThermoVL machinery beyond standard dynamical models.

---

### Polyphonic Inference

The `PolyphonicObjective/` folder contains experimental routines for “polyphonic” posterior inference, where multiple posterior voices or candidate modes can coexist rather than being immediately collapsed into one Gaussian approximation.

Key files:

* `PolyphonicObjective/fitVariationalLaplaceThermoPolyphonic.m`
* `PolyphonicObjective/fitVariationalLaplaceThermoPolyphonic_MO.m`
* `PolyphonicObjective/plotPolyphonicPosterior.m`

This is intended for exploring multimodal, non-Gaussian, or pluralistic posterior representations.

---

### Riemannian and Geometric Variants

The `Riemannian/` folder contains experimental metric-aware VL updates inspired by natural gradients and information geometry.

Key files:

* `Riemannian/fitVariationalLaplaceThermo_Riemannian.m`
* `Riemannian/defaultMetricDiagonalH.m`

---

## Repository Structure

```text
aLogLikeFit/
│
├── DEM/
│   ├── fitDEM_ThermoVL.m
│   └── dem_test/
│
├── DIP/
│   ├── dcm_moga.m
│   ├── dip_hybrid_twocmp.m
│   └── run_dip_as.m
│
├── other_experiments/
│   ├── aFitDCM.m
│   ├── fitVariationalLaplace.m
│   ├── fitVariationalLaplaceThermo_active.m
│   ├── fitVariationalLaplaceThermo_BayesARD.m
│   ├── fitVariationalLaplaceThermo_GC_EBM.m
│   ├── fitVariationalLaplaceThermo_GClam.m
│   ├── fitVariationalLaplaceThermoStable.m
│   ├── fitVariationalLaplaceThermoStruct.m
│   ├── fitVL_LowRankNoise.m
│   ├── fitVL_LowRankNoise2.m
│   ├── fitVL_ThermoMoG.m
│   ├── fitHierarchicalVL.m
│   ├── Wrapper_AlogLikeDCM_fitHierarchicalVL.m
│   └── supporting utilities and plotting functions
│
├── PEB/
│   └── peb_ard_with_stats_var.m
│
├── PEB_ARD_general/
│   ├── demo_peb_ard.m
│   ├── example_script.m
│   ├── example_usage_PEB_ARD_general.m
│   ├── peb_ard_novar.m
│   ├── peb_ard_predict.m
│   ├── peb_plot_betas.m
│   ├── peb_plot_beta_densities.m
│   ├── peb_plot_lambda.m
│   └── readme.md
│
├── PolyphonicObjective/
│   ├── fitVariationalLaplaceThermoPolyphonic.m
│   ├── fitVariationalLaplaceThermoPolyphonic_MO.m
│   └── plotPolyphonicPosterior.m
│
├── Riemannian/
│   ├── defaultMetricDiagonalH.m
│   └── fitVariationalLaplaceThermo_Riemannian.m
│
├── SVM/
│   └── fitThermoBayesSVM.m
│
├── test_fun/
│   ├── demo_fitVL_struct_toy.m
│   ├── demo_VLGC.m
│   ├── test_fitHierarchicalVL.m
│   ├── TestFun_BiExpDelay.m
│   ├── TestFun_SigmoidShift.m
│   └── TestFunc_InverseEstimatorLogisticGrowthModel.m
│
└── VL/
    ├── fitVariationalLaplaceThermo.m
    ├── fitVariationalLaplaceThermo_GC.m
    ├── fitVariationalLaplaceThermoFE.m
    ├── fitLogLikelihoodLM.m
    ├── thermoVL_equations.pdf
    ├── VL_in_GC_new.pdf
    └── VL_in_GeneralisedCoordinates.pdf
```

---

## Suggested Entry Points

For most users, the best starting points are:

| Goal                             | Suggested file                                                |
| -------------------------------- | ------------------------------------------------------------- |
| Basic thermodynamic VL           | `VL/fitVariationalLaplaceThermo.m`                            |
| VL in generalised coordinates    | `VL/fitVariationalLaplaceThermo_GC.m`                         |
| DCM-style fitting                | `other_experiments/aFitDCM.m`                                 |
| Hierarchical VL                  | `other_experiments/fitHierarchicalVL.m`                       |
| PEB / ARD analysis               | `PEB_ARD_general/demo_peb_ard.m`                              |
| Low-rank noise modelling         | `other_experiments/fitVL_LowRankNoise.m`                      |
| DEM-style inference              | `DEM/fitDEM_ThermoVL.m`                                       |
| Bayesian SVM experiment          | `SVM/fitThermoBayesSVM.m`                                     |
| Polyphonic posterior experiments | `PolyphonicObjective/fitVariationalLaplaceThermoPolyphonic.m` |
| Riemannian VL experiment         | `Riemannian/fitVariationalLaplaceThermo_Riemannian.m`         |

---

## Minimal ThermoVL Workflow

A typical workflow is:

1. Define a forward model.
2. Specify priors.
3. Run a VL or ThermoVL routine.
4. Inspect posterior estimates, free energy, and model predictions.

Example forward model:

```matlab
function yhat = my_forward_model(theta, M, U)
    % theta : parameter vector
    % M     : model structure
    % U     : inputs or experimental design
    %
    % yhat  : predicted observations

    yhat = M.fun(theta, U);
end
```

Example fitting call:

```matlab
% Observed data
y = observed_data;

% Priors
m0 = prior_mean;
S0 = prior_covariance;

% Options
OPT = struct();
OPT.Tschedule = linspace(2.0, 1.0, 16);
OPT.maxIter = 128;

% Fit model
OUT = fitVariationalLaplaceThermo(y, @my_forward_model, m0, S0, OPT);
```

Typical outputs may include:

```matlab
OUT.m      % posterior mean
OUT.S      % posterior covariance
OUT.F      % free energy trajectory
OUT.yhat   % model prediction
```

Exact output fields vary between routines.

---

## Notes on Code Status

This repository is an active methodological workspace. Some functions are mature enough for reuse, while others are exploratory prototypes.

Users should expect:

* overlapping implementations
* inconsistent function signatures across variants
* project-specific assumptions in some scripts
* partial documentation
* experimental branches of the same core idea
* occasional dependencies on local model structures or data conventions

The most stable entry points are likely to be the main routines in `VL/`, the PEB examples in `PEB_ARD_general/`, and the simpler test functions in `test_fun/`.

---

## Research Context

This code supports research on Bayesian inversion of nonlinear dynamical systems, with applications to computational psychiatry, M/EEG modelling, pharmacological neuroimaging, predictive coding, active inference, and neuro-inspired AI.

Specific methodological themes include:

* annealed variational inference
* Laplace approximations for nonlinear model fitting
* free-energy optimisation
* structured and heteroscedastic noise models
* empirical Bayes and hierarchical shrinkage
* generalised coordinates of motion
* active inference and expected free energy
* multimodal posterior approximation
* information-geometric optimisation

---

## Dependencies

The repository is written primarily in MATLAB.

Some routines may require:

* MATLAB Optimization Toolbox
* MATLAB Statistics and Machine Learning Toolbox
* SPM, for DCM-related workflows
* project-specific model files or data structures

Dependencies vary by subfolder and routine.

---

## Citation

If you use this repository in academic work, please cite the relevant paper, preprint, or repository associated with the method you use.

Suggested repository citation:

> Shaw, A. D. `aLogLikeFit`: Thermodynamic Variational Laplace and Bayesian Model Fitting Toolkit. Computational Psychiatry & Neuropharmacological Systems Lab, University of Exeter.

A formal citation will be added when associated manuscripts or archived releases are available.

---

## License

This repository is intended for academic and research use.

For commercial use, collaboration, or reuse in external projects, please contact the author.

---

## Contact

**Alexander D. Shaw**
Senior Lecturer in Neuroscience and Computational Psychiatry
University of Exeter
Website: https://cpnslab.com
