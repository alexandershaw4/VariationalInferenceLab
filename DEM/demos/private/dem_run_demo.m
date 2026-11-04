function R = dem_run_demo(name,doplot,settings)
M = dem_demo_model(name);
allowed = {'seed','noise_fraction','dt','duration','outer_iters','Dx_iters', ...
    'p','simulation_substeps','lambda_x0','lambda_x_target', ...
    'theta_true','theta0','x0_true','x0_guess','verbose','fitter_opts'};
if ~isstruct(settings) || ~isscalar(settings)
    error('DEM_demos:Settings','Settings must be a scalar structure.');
end
fields = fieldnames(settings);
for k = 1:numel(fields)
    if ~any(strcmp(fields{k},allowed))
        error('DEM_demos:Settings','Unknown setting: %s',fields{k});
    end
    M.(fields{k}) = settings.(fields{k});
end
validateattributes(M.seed,{'numeric'},{'scalar','integer','>=',0,'<=',2^32-1});
validateattributes(M.noise_fraction,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(M.dt,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(M.duration,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(M.simulation_substeps,{'numeric'},{'scalar','integer','positive'});
validateattributes(M.p,{'numeric'},{'scalar','integer','>=',1,'<=',2});
validateattributes(M.theta_true,{'numeric'},{'vector','numel',2,'real','finite','positive'});
validateattributes(M.theta0,{'numeric'},{'vector','numel',2,'real','finite','positive'});
M.theta_true = M.theta_true(:); M.theta0 = M.theta0(:);
M.x0_true = M.x0_true(:); M.x0_guess = M.x0_guess(:);
validateattributes(M.x0_true,{'numeric'},{'vector','numel',2,'real','finite'});
validateattributes(M.x0_guess,{'numeric'},{'vector','numel',2,'real','finite'});
if ~isfield(M,'verbose'), M.verbose = true; end

% Restore the caller's RNG even if the simulation or fit raises an error.
saved_rng = rng; cleanup = onCleanup(@() rng(saved_rng)); %#ok<NASGU>
rng(M.seed,'twister');
time = (0:round(M.duration/M.dt)).'*M.dt;
phi_true = log(M.theta_true); phi0 = log(M.theta0);
truth = dem_rk4(M.f,M.x0_true,phi_true,time,M.simulation_substeps);
y_true = truth(:,1);
sigma = M.noise_fraction*std(y_true);
if sigma <= eps
    error('DEM_demos:UninformativeTrajectory','The simulated observed trajectory has negligible variation.');
end
y = y_true+sigma*randn(size(y_true));

o = struct('dt',M.dt,'p',M.p,'py',0,'outer_iters',M.outer_iters, ...
    'Dx_iters',M.Dx_iters,'pE_theta',phi0,'pC_theta',diag(M.log_prior_sd.^2), ...
    'pC_x0',diag(M.gc_prior_sd(1:2*(M.p+1)).^2), ...
    'lambda_y0',1/sigma^2,'lambda_x0',M.lambda_x0, ...
    'burnin_fix_ly',3,'burnin_fix_lx',3, ...
    'lambda_x_target',M.lambda_x_target,'max_rank_theta',2, ...
    'plot',false,'verbose',M.verbose);
if isfield(M,'fitter_opts')
    extra = fieldnames(M.fitter_opts);
    for k = 1:numel(extra), o.(extra{k}) = M.fitter_opts.(extra{k}); end
end
if M.verbose, fprintf('\n=== %s | noise %.1f%% | seed %d ===\n',M.label,100*M.noise_fraction,M.seed); end
timer = tic;
[POST,TRACE] = fitDEM_ThermoVL(y,M.f,M.g,M.x0_guess,phi0,o);
elapsed = toc(timer);

R.model = rmfield(M,{'f','g'});
R.time = time; R.x_true = truth; R.y_true = y_true; R.y = y;
R.noise_sd = sigma; R.POST = POST; R.TRACE = TRACE;
R.theta_true = M.theta_true; R.theta0 = M.theta0;
R.theta_est = exp(POST.theta_mean(:));
R.yhat = POST.x_path(:,1);
R.x_free = dem_rk4(M.f,POST.x_path(1,:).',POST.theta_mean,time,M.simulation_substeps);
R.elapsed_seconds = elapsed;
R.fitter_options = o;
R.metrics.state_rmse = sqrt(mean((POST.x_path-truth).^2,1));
R.metrics.state_nrmse = R.metrics.state_rmse./max(std(truth,0,1),eps);
R.metrics.parameter_relative_error = abs(R.theta_est-M.theta_true)./M.theta_true;
R.metrics.initial_parameter_relative_error = abs(M.theta0-M.theta_true)./M.theta_true;
R.metrics.observation_rmse = sqrt(mean((R.yhat-y_true).^2));
R.metrics.data_rmse = sqrt(mean((R.yhat-y).^2));
R.metrics.free_run_nrmse = sqrt(mean((R.x_free-truth).^2,1))./max(std(truth,0,1),eps);
R.metrics.noise_sd_est = 1/sqrt(POST.lambda_y);
R.metrics.finite = all(isfinite([POST.xgc_path(:);POST.theta_mean(:); ...
    POST.theta_C(:);TRACE.Fa(:);TRACE.lambda_hist(:);R.x_free(:)]));
R.metrics.descent = all(TRACE.dstep_cost(2,:) <= TRACE.dstep_cost(1,:)+1e-8) && ...
    all(TRACE.estep_cost(2,:) <= TRACE.estep_cost(1,:)+1e-8);
R.metrics.recovery_ok = R.metrics.finite && R.metrics.descent && ...
    all(R.metrics.parameter_relative_error < M.param_error_tol) && ...
    all(R.metrics.state_nrmse < M.state_nrmse_tol);
if M.verbose
    fprintf('Parameter                 True         Initial      Estimate     Error\n');
    for k = 1:2
        fprintf('%-23s %11.5g %11.5g %11.5g %8.2f%%\n',M.theta_names{k}, ...
            M.theta_true(k),M.theta0(k),R.theta_est(k),100*R.metrics.parameter_relative_error(k));
    end
    fprintf('State NRMSE: observed %.4g, hidden %.4g | elapsed %.1fs\n', ...
        R.metrics.state_nrmse(1),R.metrics.state_nrmse(2),elapsed);
    fprintf('Noise SD: simulated %.4g, estimated %.4g | recovery check %d\n', ...
        sigma,R.metrics.noise_sd_est,R.metrics.recovery_ok);
end
if doplot, plot_DEM_demo(R); end
end
