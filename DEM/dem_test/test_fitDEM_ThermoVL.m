function test_fitDEM_ThermoVL()
% Numerical regressions for the DEM repair. No toolboxes or graphics needed.
here = fileparts(mfilename('fullpath'));
addpath(here, fileparts(here));

% A simulated trajectory and row-shaped noise must remain one time series.
signal = g_lin(zeros(2,100));
assert(isequal(size(signal + zeros(1,100)), [1 100]));

% A one-observation Gaussian update has the analytic posterior mean 4/5.
o = struct('p',0,'py',0,'outer_iters',1,'Dx_iters',8, ...
    'lambda_y0',4,'lambda_x0',1,'pC_x0',1,'pE_x0',0);
[P,R] = fitDEM_ThermoVL(1, @(x,t) 0*x, @(x,t) x, 0, 0, o);
assert(abs(P.x_path - 0.8) < 1e-5);
assert(all(isfinite(R.Fa)));

% With g independent of x, theta has a known conjugate Gaussian solution.
o.pC_theta = 100;
o.lambda_y0 = 1;
[P,R] = fitDEM_ThermoVL(2, @(x,t) 0*x, @(x,t) t(1), 0, 0, o);
assert(abs(P.theta_mean - 2/1.01) < 0.01);
assert(R.step_accepted(2,1) == 1);

% Nonlinear proposals can overflow; backtracking must find a finite descent.
o.pC_theta = 1e12;
o.theta_step = 1;
o.max_theta_step = 1e6;
[P,~] = fitDEM_ThermoVL(1000, @(x,t) 0*x, @(x,t) exp(t(1)), 0, -3, o);
assert(isfinite(P.theta_mean));
assert((exp(P.theta_mean)-1000)^2 < (exp(-3)-1000)^2);
o.theta_step = 1e4;
[P,R] = fitDEM_ThermoVL(1000, @(x,t) 0*x, @(x,t) exp(t(1)), 0, -3, o);
assert(P.theta_mean == -3 && R.step_accepted(2,1) == 0);

% Parameter recovery from a known continuous trajectory needs time coupling.
% Independent per-sample GC fitting can match y with arbitrary derivatives.
dt = .02; time = (0:80).'*dt; truth = exp(-.8*time);
o = struct('dt',dt,'p',2,'py',0,'outer_iters',4,'Dx_iters',2, ...
    'lambda_y0',1e5,'lambda_x0',10,'pC_theta',1e4, ...
    'pC_x0',diag([1e-6 1e3 1e3]),'burnin_fix_ly',10,'burnin_fix_lx',10);
[P,R] = fitDEM_ThermoVL(truth, @(x,t)-t(1)*x, @(x,t)x, 1, .3, o);
assert(abs(P.theta_mean-.8) < .01);
assert(sqrt(mean((P.xgc_path(:,2)+.8*truth).^2)) < .01);
assert(all(R.dstep_cost(2,:) <= R.dstep_cost(1,:)));
assert(all(R.estep_cost(2,:) <= R.estep_cost(1,:)));

% Multi-output observation functions may return a row vector.
o = struct('dt',0.1,'p',1,'outer_iters',2,'Dx_iters',2);
[P,R] = fitDEM_ThermoVL(zeros(10,2), @(x,t)-x, @(x,t)[x,2*x], 0, 1, o);
assert(isequal(size(P.x_path),[10 1]));
assert(all(isfinite(P.xgc_path(:))) && all(isfinite(R.lambda_hist(:))));
assert(all(eig(P.theta_V*P.theta_V.' + P.theta_D) > 0));

% Catch the original accidental T-by-T data expansion at the public API.
expect_error(@() fitDEM_ThermoVL(zeros(5), @(x,t)-x, @(x,t)x, 0, 1, o), ...
    'fitDEM_ThermoVL:ModelSize');
o.pC_theta = -1;
expect_error(@() fitDEM_ThermoVL(zeros(5,1), @(x,t)-x, @(x,t)x, 0, 1, o), ...
    'fitDEM_ThermoVL:PriorCovariance');
fprintf('test_fitDEM_ThermoVL: all numerical regressions passed.\n');
end

function expect_error(fun, identifier)
try
    fun();
catch err
    assert(strcmp(err.identifier, identifier), err.message);
    return;
end
error('Expected error %s was not raised.', identifier);
end
