# Checked demo results

These results were produced on 4 November 2026 using GNU Octave 8.4.0. Native MATLAB was not available. The three example figures were exported by the supplied MATLAB plotting function with Octave/gnuplot and visually inspected.

Each model was fitted five times: 5% observation noise with seed 42, then 3% and 10% noise with seeds 11 and 29. Noise percentages are relative to the noiseless observed trajectory SD. All 15 runs passed the numerical and recovery checks; every requested case is included.

| Model | Maximum parameter 1 error | Maximum parameter 2 error | Maximum hidden-state NRMSE | Maximum free-simulation observation NRMSE |
| --- | ---: | ---: | ---: | ---: |
| Van der Pol | 9.96% | 0.66% | 0.0392 | 0.0697 |
| FitzHugh-Nagumo | 3.28% | 1.40% | 0.0479 | 0.0883 |
| Wilson-Cowan | 0.72% | 1.42% | 0.0453 | 0.0791 |

NRMSE divides each RMSE by the corresponding true trajectory SD. A value of 0.05 therefore means an error of 5% of that state’s SD. Free simulation uses the fitted initial state and parameters without observation updates; it is an in-sample check.

Both parameter errors had to be below 15%, and both state NRMSEs below 0.25. Tests also require finite outputs, positive conditional covariance, descent of the fixed-precision D/E objectives, improved parameters relative to the guesses, and free-simulation observed NRMSE below 0.4.

The core fitter regression checks and caller RNG preservation also passed. RK4 simulation was independently compared with high-accuracy SciPy solve_ivp solutions for the default trajectories; maximum absolute state differences were below 2e-6.

The grid is a small numerical demonstration, not a statistical validation or uncertainty-coverage study. Estimated measurement-noise SD can be biased because the fitter omits posterior covariance corrections. Conditional parameter covariance also excludes state uncertainty.

To check the saved results without refitting, run from the repository root:

```matlab
addpath('DEM','DEM/demos');
load('DEM/demos/validation/checked_results.mat','checked');
test_DEM_demos(checked);
```

`recovery_summary.csv` contains one row for each of the 15 runs. `checked_results.mat` contains their complete result structures, including simulated truth, data, state estimates, traces and metrics. The PNGs show the three default 5%-noise runs.
