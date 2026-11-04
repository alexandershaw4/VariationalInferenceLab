function [POST, TRACE] = demo_fitDEM_ThermoVL(doplot)
% DEMO for fitDEM_ThermoVL on a 2-state nonlinear damped oscillator
%   x1' = x2
%   x2' = -k*x1 - c*x2 + a*tanh(x1)
%   y   = x1 + e
%
% True theta = [k; c; a]; we try to recover them from y(t).

if nargin < 1, doplot = true; end
rng(42);

% Make this demo runnable from any working directory.
demo_dir = fileparts(mfilename('fullpath'));
addpath(demo_dir, fileparts(demo_dir));

% ----- ground truth system -----
dt     = 1/200;             % 200 Hz
Tsec   = 5;                 % 5 seconds
T      = round(Tsec/dt);    % samples
nx     = 2; ny = 1;

theta_true = [ (2*pi*2)^2;   0.8*(2*pi*2);  0.6 ]; % k, c, a
x0_true    = [0.8; 0.0];

% simulate noiseless states
x = zeros(nx,T);
x(:,1) = x0_true;
for t = 2:T
    % Simulate the continuous model accurately; forward Euler introduces a
    % damping/frequency bias that should not be mistaken for fit error.
    z = x(:,t-1);
    k1 = f_osc_nl(z,theta_true);
    k2 = f_osc_nl(z+dt*k1/2,theta_true);
    k3 = f_osc_nl(z+dt*k2/2,theta_true);
    k4 = f_osc_nl(z+dt*k3,theta_true);
    x(:,t) = z+dt*(k1+2*k2+2*k3+k4)/6;
end

% generate observations
sigma_y = 0.05;                  % observation noise SD
y = g_lin(x) + sigma_y*randn(1,T);
y = y.';                         % T x ny

% ----- estimation setup -----
% initial guesses
x0_guess  = [0; 0];
theta0    = [ (2*pi*1.5)^2;  0.5*(2*pi*2);  0.0 ];

opts = struct;
opts.dt            = dt;
opts.p             = 2;       % GC order for states
opts.py            = 0;       % GC order for outputs (0th order only)
opts.Dx_iters      = 2;
opts.outer_iters   = 10;
opts.gamma_y       = 1.0;
opts.gamma_x       = 1.0;
opts.lambda_y0     = 1/sigma_y^2;
opts.lambda_x0     = 1.0;
opts.pE_theta      = theta0;
opts.pC_theta      = diag([(0.5*theta0(1))^2, (0.5*theta0(2))^2, 10^2]);
opts.pE_x0         = zeros((opts.p+1)*nx,1);
% Position, velocity and higher derivatives have different physical units.
omega = 2*pi*2;
gc_sd = [1;omega;omega;omega^2;omega^2;omega^3];
opts.pC_x0         = diag((10*gc_sd).^2);
opts.varpercthresh = 0.01;
opts.max_rank_theta= numel(theta0);
opts.plot          = doplot;
opts.verbose       = true;

opts.burnin_fix_ly = 3;

% function handles matching the required signatures:
%   f(x,theta)->nx×1, g(x,theta)->ny×1
f = @(x,theta) f_osc_nl(x,theta);
g = @(x,theta) g_lin(x);

% Keep physical units: z-scoring y alone changes the nonlinear state model.

% ----- fit -----
[POST, TRACE] = fitDEM_ThermoVL(y, f, g, x0_guess, theta0, opts);

% ----- quick report -----
fprintf('\n=== Conditional parameter estimates ===\n');
fprintf('             true        estimate     conditional SD\n');
disp([theta_true(:), POST.theta_mean(:), sqrt(diag(POST.theta_C))]);
fprintf('The small nonlinear coefficient a is weakly identified by this short damped transient.\n');

if ~doplot, return; end

% overlay fit
tspan = (0:T-1)*dt;
yhat = zeros(T,ny);
for tt = 1:T
    yhat(tt,:) = g(POST.xgc_path(tt,1:nx).', POST.theta_mean);
end

figure('Color','w','Name','Fit overlay');
plot(tspan, y, 'k-', 'DisplayName','Observed'); hold on;
plot(tspan, yhat, 'r-', 'DisplayName','Predicted');
xlabel('Time (s)'); ylabel('y'); legend; title('Observed vs Predicted');

% parameter trajectory
figure('Color','w','Name','Parameter trajectory');
names = {'k','c','a'};
for j = 1:numel(theta_true)
    subplot(3,1,j);
    plot(1:opts.outer_iters,TRACE.theta_hist(j,:),'-o'); hold on;
    plot(1:opts.outer_iters,theta_true(j)*ones(1,opts.outer_iters),'--');
    ylabel(names{j}); legend('Estimate','True','Location','best');
end
xlabel('Outer iteration');

end
