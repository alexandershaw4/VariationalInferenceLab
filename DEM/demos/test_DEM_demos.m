function results = test_DEM_demos(results)
% Scientific recovery checks. Pass existing results to avoid refitting.
%   test_DEM_demos;
%   load('DEM_results/all_results.mat'); test_DEM_demos(results);
if nargin < 1
    results = run_DEM_demos(struct('plot',false,'verbose',false));
end
if ~iscell(results), results = {results}; end
ids = cell(size(results));
for k = 1:numel(results)
    R = results{k}; ids{k} = R.model.id;
    assert(R.metrics.finite,sprintf('%s returned non-finite outputs.',R.model.id));
    assert(R.metrics.descent,sprintf('%s increased a fixed-precision objective.',R.model.id));
    assert(R.metrics.recovery_ok,sprintf( ...
        '%s failed recovery: parameter errors [%g%%,%g%%], hidden NRMSE %g.', ...
        R.model.id,100*R.metrics.parameter_relative_error(1), ...
        100*R.metrics.parameter_relative_error(2),R.metrics.state_nrmse(2)));
    assert(mean(R.metrics.parameter_relative_error) < mean(R.metrics.initial_parameter_relative_error), ...
        'Recovered parameters must improve on the initial guesses.');
    assert(R.metrics.free_run_nrmse(1) < .4, ...
        'The estimated parameters must also reproduce the observed dynamics in a free simulation.');
    assert(all(R.theta_est > 0),'Log-parameter estimates must be positive.');
    assert(all(eig(R.POST.theta_C) > 0),'Conditional log-parameter covariance must be positive definite.');
    assert(isequal(size(R.x_true),size(R.POST.x_path)),'State shapes must agree.');
    assert(isequal(size(R.y),[numel(R.time),1]),'Observations must be one time series.');
end
required = {'vanderpol','fitzhugh_nagumo','wilson_cowan'};
assert(all(ismember(required,ids)),'The check must include all three models.');
fprintf('test_DEM_demos: %d recovery runs passed.\n',numel(results));
end
