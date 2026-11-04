# Three hidden-state demonstrations for fitDEM_ThermoVL

From the repository root:

```matlab
addpath('DEM','DEM/demos');
results = run_DEM_demos;                  % three fits, three figures
```

Or run each individually. The first argument turns plotting on or off:

```matlab
vdp = demo_DEM_VanDerPol;
fhn = demo_DEM_FitzHughNagumo;
wc  = demo_DEM_WilsonCowan;
% vdp = demo_DEM_VanDerPol(false);        % headless
```

Each result includes the simulated truth, noisy observations, inferred hidden
state, parameter estimates, raw fitter outputs and recovery metrics. The six
panels show observed activity, the hidden state, phase portrait, both physical
parameter trajectories and precision updates. A free simulation starts from
the fitted initial state and uses the estimated parameters without further
observation updates. This is an in-sample dynamical check, not held-out
forecast validation.

## What the models test

All models observe only the first state. The second state and its initial
condition are inferred. Initial parameter guesses differ from the truth.
Model variables and time are dimensionless/model units.

| Model | Observed / hidden | Estimated parameters | Fixed quantities |
| --- | --- | --- | --- |
| Van der Pol | Position / velocity | Nonlinear damping mu, frequency omega | Oscillator form |
| FitzHugh-Nagumo | Voltage v / recovery w | Constant drive I, recovery timescale tau | a=0.7, b=0.8 |
| Wilson-Cowan | Excitatory E / inhibitory I | Inhibitory coupling wEI, inhibitory timescale tauI | wEE=12, wIE=10, wII=2, tauE=1, P=-1, Q=-4 |

The Van der Pol equations are:

```text
x1' = x2
x2' = mu * (1-x1^2) * x2 - omega^2 * x1
```

FitzHugh-Nagumo uses:

```text
v' = v - v^3/3 - w + I
w' = (v + 0.7 - 0.8*w) / tau
```

Wilson-Cowan uses reduced sigmoid rate equations:

```text
E' = -E + S(12*E - wEI*I - 1)
I' = (-I + S(10*E - 2*I - 4)) / tauI
S(z) = 1 / (1 + exp(-z))
```

This Wilson-Cowan variant omits refractory multipliers and baseline-shifted
sigmoids. Its fixed constants define an oscillatory example, not a fitted
biological circuit. Only the two listed parameters are free; these examples
do not establish identification of all couplings, drives and time constants.

All physical parameters are positive and fitted in log space. Consequently:

- `R.theta_est` contains physical point estimates.
- `R.POST.theta_mean` and `R.TRACE.theta_hist` are in **log-parameter space**.
- `R.POST.theta_C` is a conditional covariance in **log-parameter space**,
  excluding state uncertainty. No calibrated confidence intervals are claimed.

## Repeat across noise levels and seeds

```matlab
config = struct;
config.plot = false;
config.noise_fractions = [0.03 0.10];
config.seeds = [11 29];
config.output_dir = 'DEM_results';
[results, summary] = run_DEM_demos(config); % 3 x 2 x 2 = 12 fits
test_DEM_demos(results);                   % reuse fits; no second fitting run
```

Noise fraction means the measurement-noise SD divided by the SD of the
noiseless observed trajectory. The same setting therefore represents the
same relative observation noise across models with different amplitudes.
Observation noise is Gaussian. The underlying ODE trajectories are
deterministic: process-noise recovery is not tested. Simulation uses RK4 with
four substeps per observation interval, independently of the fitter's
finite-difference derivatives.

The simulation noise SD also initialises observation precision, after which
it is updated from residuals after three burn-in iterations. Dynamics
precision starts at 10 and is updated with a cap of 100. It regularises the
weighted collocation errors; it is not a validated estimate of a physical
process-noise variance. Posterior covariance corrections are absent from the
precision updates, so estimated observation-noise SD can be biased.

The runner retains every requested case. A recovery flag is true only when
all quantities are finite, fixed-precision D/E objectives do not increase,
both parameter relative errors are below 15%, and both state RMSEs divided
by their true trajectory SDs are below 0.25. These are transparent numerical
checks, not statistical confidence bounds. Bad recovery is reported rather
than silently omitted; numerical exceptions propagate to the caller.

No files are written unless `output_dir` is provided. When provided, it
receives one MAT file per fit, `all_results.mat`, and `summary.csv`. Optional
`config.save_plots = true` writes PNGs and needs a working graphics backend.
Reusing the same output directory replaces files with the same case names.

## Explore settings and replot

```matlab
R = demo_DEM_VanDerPol(false,struct('seed',11,'noise_fraction',.10));
fh = plot_DEM_demo(R);                    % no refit
print(fh,'vanderpol.png','-dpng','-r150');

R = demo_DEM_FitzHughNagumo(false,struct('p',1,'outer_iters',16));
```

Other overrides include `dt`, `duration`, `theta_true`, `theta0`,
`x0_true`, `x0_guess`, `simulation_substeps`, `lambda_x0`,
`lambda_x_target`, `Dx_iters`, `verbose`, and `fitter_opts`. Use the demo's
`p` option to switch GC order so the initial-state covariance dimensions
remain consistent. `fitter_opts` overrides individual fitter options last.
The caller's random-number-generator state is restored after each demo.

For tests without a saved result set:

```matlab
test_DEM_demos;                          % fits the three default cases
```

## Scope and sources

The packaged [validation results](validation/README.md) contain 15 checked
runs, saved results and example figures. All 15 passed the stated recovery
checks in GNU Octave 8.4.0. Native MATLAB was not available; the code uses
standard MATLAB syntax and no add-on optimisation toolbox.

The fitter is the repaired batch generalised-coordinate MAP/EM approximation
described in `../README.md`. These demos do not add thermodynamic annealing
or turn it into full SPM variational DEM. `TRACE.Fa` remains a negative
penalised residual energy, not a model-evidence estimate.

- [MathWorks: Van der Pol example](https://www.mathworks.com/help/matlab/math/differential-equations.html).
  This demo also estimates omega, which the standard example fixes to one.
- [FitzHugh (1961)](https://doi.org/10.1016/S0006-3495(61)86902-6).
  The equations above specify the chosen common two-variable parameterisation.
- [Wilson and Cowan (1972)](https://doi.org/10.1016/S0006-3495(72)86068-5).
  The exact reduced rate-equation variant used here is written out above.
