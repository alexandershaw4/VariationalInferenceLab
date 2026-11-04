# DEM numerical repair — 4 November 2026

For the new Van der Pol, FitzHugh-Nagumo and Wilson-Cowan demonstrations,
see [demos/README.md](demos/README.md). The new demos use this same fitter.

Run from the repository root:

```matlab
addpath('DEM', fullfile('DEM','dem_test'));
test_fitDEM_ThermoVL;
[POST, TRACE] = demo_fitDEM_ThermoVL;        % plots
% [POST, TRACE] = demo_fitDEM_ThermoVL(false); % headless
```

The public fitting signature and original output fields are retained. New
fields include the full conditional parameter covariance `POST.theta_C` and
before/after costs `TRACE.dstep_cost` and `TRACE.estep_cost`.

## Causes of the failure

- `g_lin(x)` returned a T-by-1 trajectory, but the demo added 1-by-T noise.
  Implicit expansion produced a T-by-T observation matrix: 1,000,000 values
  instead of 1,000. It now returns ny-by-T trajectories.
- Both recognition and parameter steps added `J'*r` while adding the step to
  the estimate. For least squares the descent right-hand side is `-J'*r`.
- Backoff accepted its last trial even when the residual grew, and did not
  reject non-finite trials. The replacement accepts only finite decreases in
  the complete objective, including priors, or retains the current estimate.
- Each sample could fit its own generalised state independently of the time
  derivatives of its neighbours. A finite fit was therefore possible without
  learning the dynamics. Generalised states now derive from one trajectory
  using sparse time-derivative operators, and the D-step smooths that whole
  trajectory. A joint state/parameter Gauss-Newton step accelerates learning
  when a hidden velocity can compensate for wrong parameters.
- Truncating the shift operator forced the unavailable highest motion
  derivative to zero. Motion residuals now stop at order p-1. State order p=2
  represents x, x', x'' and constrains x'=f and x''=J_f*x', without imposing
  zero jerk. Higher-order residuals are scaled by powers of dt; the actual
  zeroth-order flow error retains unit weight.
- The demo standardised y without transforming the nonlinear state model.
  It now fits in physical units, uses initial-state prior scales appropriate
  to each derivative's units, and simulates the continuous model with RK4.
- The returned parameter curvature is recomputed at the final parameters and
  precisions. Optimisation damping is excluded from posterior precision.

## Interpretation

This repair replaces the disconnected per-sample recognition update with a
batch generalised-coordinate MAP smoother and joint Gauss-Newton parameter
updates. It is an algorithmic change, not just NaN suppression. It supports
state orders 0–2, output orders 0–p, and requires at least p+1 samples.

It remains a **DEM-style MAP/EM approximation**, rather than the full SPM DEM
variational filtering algorithm. The scalar precision updates omit posterior
covariance corrections; derivative error correlations and temporal noise
covariance are not estimated. There is no thermodynamic temperature schedule.
`TRACE.Fa` is negative penalised residual energy, **not an ELBO or model
evidence**, and need not be monotone when precisions change.

`POST.theta_V*POST.theta_V' + POST.theta_D` is an approximation to parameter
precision. `POST.theta_C` is the inverse full Gauss-Newton precision
conditional on the fitted trajectory; it does not marginalise state
uncertainty. Treat these conditional uncertainty estimates accordingly.

The original oscillator's nonlinear coefficient a=0.6 has very little effect
compared with k≈158, especially as the transient rapidly decays. This demo
does not establish separate recovery of all three parameters. A stronger
identification study would need more informative dynamics, repeated
trajectories or external input, and sensitivity/prior checks.

## Validation

The original oscillator repair was executed in GNU Octave 8.4.0 with plotting
disabled. Native MATLAB and the fitter's live progress plots were not
executed. The later demos have separately verified figure exports; see their
validation notes. No add-on optimisation toolbox is required.

The original solver failed with `svd: cannot take SVD of matrix containing
Inf or NaN values` on the 1,000-sample demo even after correcting its input
shape. Numerical regression checks cover:

- Correct simulated-signal shape.
- Analytic Gaussian state and parameter posterior means.
- Overshooting/overflowing trial steps and complete step rejection.
- Recovery of a known decay constant and its trajectory derivatives.
- Non-increasing D/E objectives at fixed precisions.
- Multiple observation outputs, including row-returning observation functions.
- Positive conditional precision and explicit rejection of malformed data
  dimensions or invalid prior covariances.

The full 1,000-sample, ten-iteration oscillator run is checked separately for
finite states, parameters, precisions and cost traces, and for accepted D/E
steps that do not increase their fixed-precision objectives.

With seed 42, RK4 simulation, 200 Hz and 0.05 observation-noise SD, the checked
run gave the following point estimates. These are a single demonstration,
not a coverage or identifiability study:

| Parameter | True | Estimate |
| --- | ---: | ---: |
| k | 157.9137 | 156.19 |
| c | 10.0531 | 9.8444 |
| a | 0.6000 | -3.6909 |

The run took approximately 207 seconds in this Octave environment. The
observation precision was 437.85, compared with the simulated value of 400.
All states, parameter estimates, precisions and cost traces were finite.
The nonlinear coefficient is **not successfully recovered** in this example.
