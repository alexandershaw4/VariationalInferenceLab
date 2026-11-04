function M = dem_demo_model(name)
% All estimated physical parameters are positive and fitted in log space.
% Fixed constants are explicit: these are two-parameter identification demos.
M.id = name; M.g = @(x,phi) x(1); %#ok<INUSD>
M.p = 2; M.outer_iters = 12; M.Dx_iters = 1;
M.noise_fraction = .05; M.seed = 42; M.simulation_substeps = 4;
M.lambda_x0 = 10; M.lambda_x_target = 100;
M.param_error_tol = .15; M.state_nrmse_tol = .25;
switch name
    case 'vanderpol'
        M.label = 'Van der Pol';
        M.f = @vanderpol_flow;
        M.dt = .08; M.duration = 19.2;
        M.theta_true = [1;1]; M.theta0 = [.6;1.15];
        M.theta_names = {'mu','omega'}; M.log_prior_sd = [.6;.4];
        M.x0_true = [1.5;0]; M.x0_guess = [0;0];
        M.gc_prior_sd = [5;5;10;10;20;20];
        M.state_names = {'Position','Velocity'};
        M.description = 'Position observed; velocity hidden. Both mu and omega are estimated.';
        M.reference = 'https://www.mathworks.com/help/matlab/math/differential-equations.html';
    case 'fitzhugh_nagumo'
        M.label = 'FitzHugh-Nagumo';
        M.f = @fitzhugh_flow;
        M.dt = .2; M.duration = 60;
        M.theta_true = [.6;8]; M.theta0 = [.4;10.5];
        M.theta_names = {'Drive I','Recovery tau'}; M.log_prior_sd = [.6;.5];
        M.x0_true = [-1;-.2]; M.x0_guess = [0;0];
        M.gc_prior_sd = [5;3;5;3;10;5];
        M.state_names = {'Voltage v','Recovery w'};
        M.description = 'Voltage observed; recovery hidden. Fixed a=.7, b=.8.';
        M.reference = 'https://doi.org/10.1016/S0006-3495(61)86902-6';
    case 'wilson_cowan'
        M.label = 'Wilson-Cowan';
        M.f = @wilson_cowan_flow;
        M.dt = .16; M.duration = 40;
        M.theta_true = [10;2]; M.theta0 = [8.5;1.5];
        M.theta_names = {'Inhibition wEI','Inhibitory tauI'}; M.log_prior_sd = [.35;.4];
        M.x0_true = [.2;.3]; M.x0_guess = [.5;.5];
        M.gc_prior_sd = [2;2;2;2;4;4];
        M.state_names = {'Excitatory E','Inhibitory I'};
        M.description = 'E observed; I hidden. Other couplings, drives and tauE fixed.';
        M.reference = 'https://doi.org/10.1016/S0006-3495(72)86068-5';
    otherwise
        error('DEM_demos:UnknownModel','Unknown demo model: %s',name);
end
end

function dx = vanderpol_flow(x,phi)
mu = exp(phi(1)); omega = exp(phi(2));
dx = [x(2);mu*(1-x(1)^2)*x(2)-omega^2*x(1)];
end

function dx = fitzhugh_flow(x,phi)
drive = exp(phi(1)); tau = exp(phi(2));
v = x(1); w = x(2); a = .7; b = .8;
dx = [v-v^3/3-w+drive;(v+a-b*w)/tau];
end

function dx = wilson_cowan_flow(x,phi)
wEI = exp(phi(1)); tauI = exp(phi(2));
E = x(1); I = x(2);
% Reduced rate equations, without a refractory multiplier or baseline shift.
% Stable sigmoid evaluation also keeps rejected line-search trials finite.
S = @(z) .5*(1+tanh(z/2));
wEE = 12; wIE = 10; wII = 2; tauE = 1; P = -1; Q = -4;
dx = [(-E+S(wEE*E-wEI*I+P))/tauE; ...
      (-I+S(wIE*E-wII*I+Q))/tauI];
end
