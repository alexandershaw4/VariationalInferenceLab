function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs, free_energy, ...
          beta_sd, Pp_pos, Pp_sign, robustness] = ...
    peb_ard_with_stats_var(theta, X, Sigma_theta_prior, max_iter, tol, opts)
%PEB_ARD_WITH_STATS_VAR Parametric empirical Bayes with ARD and optional robustness checks.
%
% This is a backwards-compatible extension of the original routine. The first
% ten outputs are unchanged. If requested, an eleventh output, ROBUSTNESS, is
% returned as a struct containing leave-one-subject-out, bootstrap and partial
% regression diagnostics for effects with high posterior sign probability.
%
% Usage, original behaviour:
%   [b,l,t,p,pm,pc,F,bsd,Ppos,Psign] = peb_ard_with_stats_var(theta,X,Sigma,100,1e-6);
%
% Usage, with robustness checks:
%   opts = struct;
%   opts.robustness = true;
%   opts.robust_threshold = 0.99;     % effects to check: Pp_sign >= threshold
%   opts.n_boot = 500;               % bootstrap refits
%   opts.methods = {'loo','bootstrap','partial'};
%   [b,l,t,p,pm,pc,F,bsd,Ppos,Psign,R] = peb_ard_with_stats_var(theta,X,Sigma,100,1e-6,opts);
%
% Inputs:
% - theta: Individual-level parameters (N x d)
% - X: Design matrix (N x p)
% - Sigma_theta_prior: Prior covariance of individual-level parameters (N x d x d)
% - max_iter: Maximum number of iterations
% - tol: Convergence tolerance
% - opts: Optional struct or logical. If true, enables default robustness checks.
%
% Additional opts fields:
% - robustness: true/false, run robustness checks. Default false.
% - robust_threshold: posterior sign probability threshold for effects to check. Default 0.99.
% - selected_effects: optional p x d logical mask or linear indices. Default Pp_sign >= threshold.
% - methods: cell array containing any of {'loo','bootstrap','partial'}. Default all three.
% - n_boot: number of bootstrap refits. Default 250.
% - alpha: bootstrap interval alpha. Default 0.05.
% - min_sign_stability: required sign stability for robust_mask_sign. Default 0.95.
% - loo_min_Pp_sign: required minimum LOO Pp_sign for robust_mask_sign. Default 0.95.
% - cook_threshold: Cook's distance threshold for influence diagnostics. Default 4/N.
% - studentized_threshold: abs studentized residual threshold. Default 3.
% - store_full: store full LOO/bootstrap beta arrays. Default false.
% - store_partial_residuals: store x/y partial residuals for selected effects. Default true.
% - predictor_names: optional p x 1 cellstr for readable summary tables.
% - param_names: optional d x 1 cellstr for readable summary tables.
% - subject_ids: optional N x 1 labels used in selected-effect diagnostics.
% - rng_seed: optional seed for reproducible bootstrap.
% - verbose: print convergence messages. Default true for main fit, false for refits.
%
% Robustness struct fields:
% - selected_mask: p x d logical matrix of effects checked.
% - loo: leave-one-subject-out summaries.
% - bootstrap: non-parametric subject bootstrap summaries.
% - partial: partial-regression scatter/influence diagnostics.
% - robust_mask_sign: effects with stable sign across LOO/bootstrap.
% - robust_mask_conservative: robust_mask_sign plus no flagged influential subjects.
% - summary: table with one row per selected effect.
%
% AS2024; robustness extension AS2026.

if nargin < 4 || isempty(max_iter)
    max_iter = 100;
end
if nargin < 5 || isempty(tol)
    tol = 1e-6;
end
if nargin < 6 || isempty(opts)
    opts = struct();
end
opts = local_parse_opts(opts, max_iter, tol);

[beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs, free_energy, ...
    beta_sd, Pp_pos, Pp_sign, core] = local_peb_core(theta, X, Sigma_theta_prior, max_iter, tol, opts.verbose);

robustness = struct();
robustness.enabled = false;
if opts.robustness
    robustness = local_robustness(theta, X, Sigma_theta_prior, max_iter, tol, ...
        beta, beta_sd, Pp_pos, Pp_sign, opts, core);
end

end

% ========================================================================
% Core PEB routine: original algorithm, factored out so robustness checks can
% refit the model without recursively calling the public function.
% ========================================================================
function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs, free_energy, ...
          beta_sd, Pp_pos, Pp_sign, core] = local_peb_core(theta, X, Sigma_theta_prior, max_iter, tol, verbose)

[N, d] = size(theta);
p = size(X, 2);

lambda_vals = ones(p, 1);
beta = zeros(p, d);
sigma_squared = ones(N, d);

Sigma_theta_prior_inv = zeros(N, d, d);
for i = 1:N
    Sigma_theta_prior_inv(i, :, :) = inv(squeeze(Sigma_theta_prior(i, :, :)) + 1e-6 * eye(d));
end

converged = false;
iter_used = max_iter;
for iter = 1:max_iter
    beta_old = beta;

    for j = 1:p
        beta(j, :) = (X(:, j)' * (theta - X * beta + X(:, j) * beta(j, :))) ./ ...
            (X(:, j)' * X(:, j) + lambda_vals(j));
    end

    lambda_vals = (1 + 1e-6) ./ (sum(beta.^2, 2) + 1e-6);
    lambda_vals = min(lambda_vals, 1e6);
    lambda_vals = max(lambda_vals, 1e-6);

    for i = 1:N
        residuals_i = (theta(i, :) - (X(i, :) * beta))';
        sigma_squared(i, :) = (residuals_i.^2 + diag(squeeze(Sigma_theta_prior(i, :, :)))) / 2;
    end

    if norm(beta - beta_old, 'fro') < tol
        converged = true;
        iter_used = iter;
        if verbose
            fprintf('Converged after %d iterations.\n', iter);
        end
        break;
    end
end

if ~converged && verbose
    fprintf('Did not converge after %d iterations.\n', max_iter);
end

posterior_means = zeros(N, d);
posterior_covs = zeros(N, d, d);

for i = 1:N
    Sigma_post_inv = squeeze(Sigma_theta_prior_inv(i, :, :)) + diag(1 ./ sigma_squared(i, :));
    posterior_covs(i, :, :) = inv(Sigma_post_inv);

    posterior_means(i, :) = (squeeze(posterior_covs(i, :, :)) * ...
        (squeeze(Sigma_theta_prior_inv(i, :, :)) * theta(i, :)' + ...
        diag(1 ./ sigma_squared(i, :)) * (X(i, :) * beta)'))';
end

sigma2 = mean(sigma_squared, 1);
XtX = X' * X;
Lambda = diag(lambda_vals);
A = XtX + Lambda;
A = 0.5 * (A + A') + 1e-10 * eye(p);
Ainv = inv(A);

beta_var = zeros(p, d);
for k = 1:d
    beta_var(:, k) = sigma2(k) * diag(Ainv);
end
beta_sd = sqrt(max(beta_var, 1e-18));

t_stats = beta ./ max(beta_sd, 1e-12);
df = max(N - p, 1);
p_values = 2 * (1 - tcdf(abs(t_stats), df));

Pp_pos  = 1 - normcdf(0, beta, beta_sd);
Pp_sign = max(Pp_pos, 1 - Pp_pos);

log2pi = log(2 * pi);
F = 0;
for i = 1:N
    mu_q = posterior_means(i, :)';
    Sigma_q = squeeze(posterior_covs(i, :, :));
    mu_p = (X(i, :) * beta)';

    KL = 0.5 * (trace(squeeze(Sigma_theta_prior_inv(i, :, :)) * Sigma_q) + ...
        (theta(i, :)' - mu_q)' * squeeze(Sigma_theta_prior_inv(i, :, :)) * (theta(i, :)' - mu_q) - ...
        d + log(det(squeeze(Sigma_theta_prior(i, :, :))) + 1e-10) - log(det(Sigma_q) + 1e-10));

    diff = theta(i, :)' - mu_p;
    ELL = -0.5 * (d * log2pi + sum(log(sigma_squared(i, :))) + sum((diff.^2) ./ sigma_squared(i, :)'));

    F = F + ELL - KL;
end
free_energy = F;

core = struct();
core.sigma_squared = sigma_squared;
core.sigma2 = sigma2;
core.Ainv = Ainv;
core.iter_used = iter_used;
core.converged = converged;

end

% ========================================================================
% Robustness diagnostics
% ========================================================================
function robustness = local_robustness(theta, X, Sigma_theta_prior, max_iter, tol, ...
    beta, beta_sd, Pp_pos, Pp_sign, opts, core)

[N, d] = size(theta);
p = size(X, 2);

selected = local_selected_mask(opts.selected_effects, Pp_sign, opts.robust_threshold, p, d);
if isempty(opts.cook_threshold)
    opts.cook_threshold = 4 / N;
end
max_iter = opts.refit_max_iter;
tol = opts.refit_tol;
base_sign = local_effect_sign(beta, Pp_pos);

robustness = struct();
robustness.enabled = true;
robustness.threshold = opts.robust_threshold;
robustness.methods = opts.methods;
robustness.selected_mask = selected;
robustness.n_selected = nnz(selected);
robustness.core = core;
robustness.note = ['robust_mask_sign requires stable effect direction; ', ...
    'robust_mask_conservative also requires no flagged influential subjects in the partial-regression diagnostic.'];

if nnz(selected) == 0
    robustness.summary = table();
    robustness.robust_mask_sign = false(p, d);
    robustness.robust_mask_conservative = false(p, d);
    return;
end

if local_has_method(opts.methods, 'loo')
    robustness.loo = local_leave_one_out(theta, X, Sigma_theta_prior, max_iter, tol, ...
        beta, Pp_pos, selected, base_sign, opts);
else
    robustness.loo = [];
end

if local_has_method(opts.methods, 'bootstrap')
    robustness.bootstrap = local_bootstrap(theta, X, Sigma_theta_prior, max_iter, tol, ...
        beta, Pp_pos, selected, base_sign, opts);
else
    robustness.bootstrap = [];
end

if local_has_method(opts.methods, 'partial')
    robustness.partial = local_partial_diagnostics(theta, X, beta, beta_sd, Pp_sign, selected, opts);
else
    robustness.partial = [];
end

[robustness.robust_mask_sign, robustness.robust_mask_conservative, robustness.summary] = ...
    local_robust_masks_and_summary(beta, beta_sd, Pp_sign, selected, robustness, opts);

end

function loo = local_leave_one_out(theta, X, Sigma_theta_prior, max_iter, tol, beta, Pp_pos, selected, base_sign, opts)

[N, d] = size(theta);
p = size(X, 2);

beta_loo = nan(p, d, N);
Pp_pos_loo = nan(p, d, N);
Pp_sign_loo = nan(p, d, N);

for ii = 1:N
    keep = true(N, 1);
    keep(ii) = false;
    [b_i, ~, ~, ~, ~, ~, ~, ~, Ppos_i, Psign_i] = ...
        local_peb_core(theta(keep, :), X(keep, :), Sigma_theta_prior(keep, :, :), max_iter, tol, false);
    beta_loo(:, :, ii) = b_i;
    Pp_pos_loo(:, :, ii) = Ppos_i;
    Pp_sign_loo(:, :, ii) = Psign_i;
end

sign_match = false(p, d, N);
for ii = 1:N
    sign_i = local_effect_sign(beta_loo(:, :, ii), Pp_pos_loo(:, :, ii));
    sign_match(:, :, ii) = sign_i == base_sign;
end

delta = abs(beta_loo - repmat(beta, [1 1 N]));
rel_delta = delta ./ max(abs(repmat(beta, [1 1 N])), 1e-12);

[loo.max_abs_delta, loo.worst_subject_abs_delta] = max(delta, [], 3);
[loo.max_rel_delta, loo.worst_subject_rel_delta] = max(rel_delta, [], 3);
loo.min_Pp_sign = min(Pp_sign_loo, [], 3);
loo.mean_Pp_sign = mean(Pp_sign_loo, 3);
loo.sign_stability = mean(sign_match, 3);
loo.beta_mean = mean(beta_loo, 3);
loo.beta_sd = std(beta_loo, 0, 3);

loo.max_abs_delta = local_mask_nan(loo.max_abs_delta, selected);
loo.worst_subject_abs_delta = local_mask_nan(loo.worst_subject_abs_delta, selected);
loo.max_rel_delta = local_mask_nan(loo.max_rel_delta, selected);
loo.worst_subject_rel_delta = local_mask_nan(loo.worst_subject_rel_delta, selected);
loo.min_Pp_sign = local_mask_nan(loo.min_Pp_sign, selected);
loo.mean_Pp_sign = local_mask_nan(loo.mean_Pp_sign, selected);
loo.sign_stability = local_mask_nan(loo.sign_stability, selected);
loo.beta_mean = local_mask_nan(loo.beta_mean, selected);
loo.beta_sd = local_mask_nan(loo.beta_sd, selected);

if opts.store_full
    loo.beta = beta_loo;
    loo.Pp_pos = Pp_pos_loo;
    loo.Pp_sign = Pp_sign_loo;
end

end

function boot = local_bootstrap(theta, X, Sigma_theta_prior, max_iter, tol, beta, Pp_pos, selected, base_sign, opts)

[N, d] = size(theta);
p = size(X, 2);
B = opts.n_boot;

if ~isempty(opts.rng_seed)
    rng(opts.rng_seed);
end

beta_boot = nan(p, d, B);
for bb = 1:B
    idx = randi(N, [N, 1]);
    [b_b, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
        local_peb_core(theta(idx, :), X(idx, :), Sigma_theta_prior(idx, :, :), max_iter, tol, false);
    beta_boot(:, :, bb) = b_b;
end

boot.beta_mean = mean(beta_boot, 3);
boot.beta_sd = std(beta_boot, 0, 3);
boot.ci_low = prctile(beta_boot, 100 * opts.alpha / 2, 3);
boot.ci_high = prctile(beta_boot, 100 * (1 - opts.alpha / 2), 3);
boot.pr_pos = mean(beta_boot > 0, 3);
boot.pr_base_sign = boot.pr_pos;
neg = base_sign < 0;
boot.pr_base_sign(neg) = 1 - boot.pr_pos(neg);
boot.ci_excludes_zero = (boot.ci_low > 0 & boot.ci_high > 0) | (boot.ci_low < 0 & boot.ci_high < 0);

boot.beta_mean = local_mask_nan(boot.beta_mean, selected);
boot.beta_sd = local_mask_nan(boot.beta_sd, selected);
boot.ci_low = local_mask_nan(boot.ci_low, selected);
boot.ci_high = local_mask_nan(boot.ci_high, selected);
boot.pr_pos = local_mask_nan(boot.pr_pos, selected);
boot.pr_base_sign = local_mask_nan(boot.pr_base_sign, selected);
boot.ci_excludes_zero = local_mask_false(boot.ci_excludes_zero, selected);

if opts.store_full
    boot.beta = beta_boot;
end

end

function partial = local_partial_diagnostics(theta, X, beta, beta_sd, Pp_sign, selected, opts)

[N, d] = size(theta);
p = size(X, 2);

partial.slope = nan(p, d);
partial.slope_se = nan(p, d);
partial.t_stat = nan(p, d);
partial.p_value = nan(p, d);
partial.r_pearson = nan(p, d);
partial.r_spearman = nan(p, d);
partial.max_cook = nan(p, d);
partial.n_cook_over_threshold = nan(p, d);
partial.n_abs_studentized_over_threshold = nan(p, d);
partial.n_influential = nan(p, d);
partial.cook_threshold = opts.cook_threshold;
partial.studentized_threshold = opts.studentized_threshold;
partial.effects = struct([]);

effect_count = 0;
for j = 1:p
    others = setdiff(1:p, j);
    x_resid = local_residualize(X(:, j), X(:, others));

    for k = 1:d
        y_resid = local_residualize(theta(:, k), X(:, others));
        stats = local_simple_regression_diagnostics(x_resid, y_resid, opts.cook_threshold, opts.studentized_threshold);

        partial.slope(j, k) = stats.slope;
        partial.slope_se(j, k) = stats.slope_se;
        partial.t_stat(j, k) = stats.t_stat;
        partial.p_value(j, k) = stats.p_value;
        partial.r_pearson(j, k) = stats.r_pearson;
        partial.r_spearman(j, k) = stats.r_spearman;
        partial.max_cook(j, k) = stats.max_cook;
        partial.n_cook_over_threshold(j, k) = stats.n_cook_over_threshold;
        partial.n_abs_studentized_over_threshold(j, k) = stats.n_abs_studentized_over_threshold;
        partial.n_influential(j, k) = numel(stats.influential_subjects);

        if selected(j, k)
            effect_count = effect_count + 1;
            partial.effects(effect_count).predictor_index = j;
            partial.effects(effect_count).parameter_index = k;
            partial.effects(effect_count).predictor_name = local_name(opts.predictor_names, j, 'X');
            partial.effects(effect_count).parameter_name = local_name(opts.param_names, k, 'param');
            partial.effects(effect_count).beta = beta(j, k);
            partial.effects(effect_count).beta_sd = beta_sd(j, k);
            partial.effects(effect_count).Pp_sign = Pp_sign(j, k);
            partial.effects(effect_count).partial_slope = stats.slope;
            partial.effects(effect_count).partial_r_pearson = stats.r_pearson;
            partial.effects(effect_count).partial_r_spearman = stats.r_spearman;
            partial.effects(effect_count).max_cook = stats.max_cook;
            partial.effects(effect_count).influential_subjects = stats.influential_subjects;
            if ~isempty(opts.subject_ids)
                partial.effects(effect_count).influential_subject_ids = opts.subject_ids(stats.influential_subjects);
            end
            if opts.store_partial_residuals
                partial.effects(effect_count).x_resid = stats.x;
                partial.effects(effect_count).y_resid = stats.y;
                partial.effects(effect_count).fitted = stats.fitted;
                partial.effects(effect_count).residual = stats.residual;
                partial.effects(effect_count).cooks_distance = stats.cooks_distance;
                partial.effects(effect_count).studentized_residual = stats.studentized_residual;
            end
        end
    end
end

partial.slope = local_mask_nan(partial.slope, selected);
partial.slope_se = local_mask_nan(partial.slope_se, selected);
partial.t_stat = local_mask_nan(partial.t_stat, selected);
partial.p_value = local_mask_nan(partial.p_value, selected);
partial.r_pearson = local_mask_nan(partial.r_pearson, selected);
partial.r_spearman = local_mask_nan(partial.r_spearman, selected);
partial.max_cook = local_mask_nan(partial.max_cook, selected);
partial.n_cook_over_threshold = local_mask_nan(partial.n_cook_over_threshold, selected);
partial.n_abs_studentized_over_threshold = local_mask_nan(partial.n_abs_studentized_over_threshold, selected);
partial.n_influential = local_mask_nan(partial.n_influential, selected);

end

function stats = local_simple_regression_diagnostics(x, y, cook_threshold, studentized_threshold)

x = x(:);
y = y(:);
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);
n = numel(x);

stats = struct();
stats.x = x;
stats.y = y;
stats.slope = NaN;
stats.slope_se = NaN;
stats.t_stat = NaN;
stats.p_value = NaN;
stats.r_pearson = NaN;
stats.r_spearman = NaN;
stats.max_cook = NaN;
stats.n_cook_over_threshold = NaN;
stats.n_abs_studentized_over_threshold = NaN;
stats.influential_subjects = [];
stats.fitted = NaN(size(y));
stats.residual = NaN(size(y));
stats.cooks_distance = NaN(size(y));
stats.studentized_residual = NaN(size(y));

if n < 3 || std(x) < eps || std(y) < eps
    return;
end

Xr = [ones(n, 1), x];
b = Xr \ y;
yhat = Xr * b;
r = y - yhat;
df = max(n - 2, 1);
SSE = sum(r .^ 2);
MSE = SSE / df;
CovB = MSE * pinv(Xr' * Xr);
se = sqrt(max(diag(CovB), 0));
H = Xr * pinv(Xr' * Xr) * Xr';
h = min(max(diag(H), 0), 1 - 1e-12);
studentized = r ./ sqrt(max(MSE * (1 - h), eps));
cooks = (r .^ 2 ./ max(2 * MSE, eps)) .* (h ./ max((1 - h) .^ 2, eps));

stats.slope = b(2);
stats.slope_se = se(2);
stats.t_stat = b(2) / max(se(2), 1e-12);
stats.p_value = 2 * (1 - tcdf(abs(stats.t_stat), df));
stats.r_pearson = corr(x, y, 'type', 'Pearson', 'rows', 'complete');
stats.r_spearman = corr(x, y, 'type', 'Spearman', 'rows', 'complete');
stats.max_cook = max(cooks);
stats.n_cook_over_threshold = sum(cooks > cook_threshold);
stats.n_abs_studentized_over_threshold = sum(abs(studentized) > studentized_threshold);
stats.influential_subjects = find(cooks > cook_threshold | abs(studentized) > studentized_threshold);
stats.fitted = yhat;
stats.residual = r;
stats.cooks_distance = cooks;
stats.studentized_residual = studentized;

end

function y_resid = local_residualize(y, Z)

y = y(:);
if isempty(Z)
    y_resid = y;
    return;
end
Z = double(Z);
valid_cols = any(abs(Z) > eps, 1);
Z = Z(:, valid_cols);
if isempty(Z)
    y_resid = y;
else
    y_resid = y - Z * (pinv(Z) * y);
end

end

function [robust_mask_sign, robust_mask_conservative, summary] = ...
    local_robust_masks_and_summary(beta, beta_sd, Pp_sign, selected, robustness, opts)

[p, d] = size(beta);
robust_mask_sign = selected;
robust_mask_conservative = selected;

if isstruct(robustness.loo)
    robust_mask_sign = robust_mask_sign & ...
        robustness.loo.sign_stability >= opts.min_sign_stability & ...
        robustness.loo.min_Pp_sign >= opts.loo_min_Pp_sign;
end

if isstruct(robustness.bootstrap)
    robust_mask_sign = robust_mask_sign & ...
        robustness.bootstrap.pr_base_sign >= opts.min_sign_stability & ...
        robustness.bootstrap.ci_excludes_zero;
end

robust_mask_conservative = robust_mask_sign;
if isstruct(robustness.partial)
    robust_mask_conservative = robust_mask_conservative & ...
        robustness.partial.n_influential == 0;
end

[idx_j, idx_k] = find(selected);
nE = numel(idx_j);
PredictorIndex = idx_j;
ParameterIndex = idx_k;
Predictor = cell(nE, 1);
Parameter = cell(nE, 1);
Beta = nan(nE, 1);
BetaSD = nan(nE, 1);
PpSign = nan(nE, 1);
LOO_MinPpSign = nan(nE, 1);
LOO_SignStability = nan(nE, 1);
LOO_MaxAbsDelta = nan(nE, 1);
LOO_WorstSubject = nan(nE, 1);
Boot_PrBaseSign = nan(nE, 1);
Boot_CILow = nan(nE, 1);
Boot_CIHigh = nan(nE, 1);
Boot_CIExcludesZero = false(nE, 1);
Partial_RPearson = nan(nE, 1);
Partial_RSpearman = nan(nE, 1);
Partial_MaxCook = nan(nE, 1);
Partial_NInfluential = nan(nE, 1);
RobustSign = false(nE, 1);
RobustConservative = false(nE, 1);

for ee = 1:nE
    j = idx_j(ee);
    k = idx_k(ee);
    Predictor{ee} = local_name(opts.predictor_names, j, 'X');
    Parameter{ee} = local_name(opts.param_names, k, 'param');
    Beta(ee) = beta(j, k);
    BetaSD(ee) = beta_sd(j, k);
    PpSign(ee) = Pp_sign(j, k);

    if isstruct(robustness.loo)
        LOO_MinPpSign(ee) = robustness.loo.min_Pp_sign(j, k);
        LOO_SignStability(ee) = robustness.loo.sign_stability(j, k);
        LOO_MaxAbsDelta(ee) = robustness.loo.max_abs_delta(j, k);
        LOO_WorstSubject(ee) = robustness.loo.worst_subject_abs_delta(j, k);
    end

    if isstruct(robustness.bootstrap)
        Boot_PrBaseSign(ee) = robustness.bootstrap.pr_base_sign(j, k);
        Boot_CILow(ee) = robustness.bootstrap.ci_low(j, k);
        Boot_CIHigh(ee) = robustness.bootstrap.ci_high(j, k);
        Boot_CIExcludesZero(ee) = robustness.bootstrap.ci_excludes_zero(j, k);
    end

    if isstruct(robustness.partial)
        Partial_RPearson(ee) = robustness.partial.r_pearson(j, k);
        Partial_RSpearman(ee) = robustness.partial.r_spearman(j, k);
        Partial_MaxCook(ee) = robustness.partial.max_cook(j, k);
        Partial_NInfluential(ee) = robustness.partial.n_influential(j, k);
    end

    RobustSign(ee) = robust_mask_sign(j, k);
    RobustConservative(ee) = robust_mask_conservative(j, k);
end

summary = table(PredictorIndex, ParameterIndex, Predictor, Parameter, Beta, BetaSD, PpSign, ...
    LOO_MinPpSign, LOO_SignStability, LOO_MaxAbsDelta, LOO_WorstSubject, ...
    Boot_PrBaseSign, Boot_CILow, Boot_CIHigh, Boot_CIExcludesZero, ...
    Partial_RPearson, Partial_RSpearman, Partial_MaxCook, Partial_NInfluential, ...
    RobustSign, RobustConservative);

if ~isempty(summary)
    summary = sortrows(summary, {'RobustConservative','PpSign'}, {'descend','descend'});
end

end

% ========================================================================
% Helpers
% ========================================================================
function opts = local_parse_opts(opts, max_iter, tol)

if islogical(opts) || isnumeric(opts)
    opts = struct('robustness', logical(opts));
end

if isfield(opts, 'robust') && ~isfield(opts, 'robustness')
    opts.robustness = opts.robust;
end

opts = local_default(opts, 'robustness', false);
opts = local_default(opts, 'robust_threshold', 0.99);
opts = local_default(opts, 'selected_effects', []);
opts = local_default(opts, 'methods', {'loo','bootstrap','partial'});
opts = local_default(opts, 'n_boot', 250);
opts = local_default(opts, 'alpha', 0.05);
opts = local_default(opts, 'min_sign_stability', 0.95);
opts = local_default(opts, 'loo_min_Pp_sign', 0.95);
opts = local_default(opts, 'cook_threshold', []);
opts = local_default(opts, 'studentized_threshold', 3);
opts = local_default(opts, 'store_full', false);
opts = local_default(opts, 'store_partial_residuals', true);
opts = local_default(opts, 'predictor_names', {});
opts = local_default(opts, 'param_names', {});
opts = local_default(opts, 'subject_ids', []);
opts = local_default(opts, 'rng_seed', []);
opts = local_default(opts, 'verbose', true);
opts = local_default(opts, 'refit_max_iter', max_iter);
opts = local_default(opts, 'refit_tol', tol);

if ischar(opts.methods)
    opts.methods = {opts.methods};
end
if isempty(opts.n_boot) || opts.n_boot < 1
    opts.n_boot = 250;
end
if isempty(opts.alpha) || opts.alpha <= 0 || opts.alpha >= 1
    opts.alpha = 0.05;
end

end

function opts = local_default(opts, field, value)
if ~isfield(opts, field) || isempty(opts.(field))
    opts.(field) = value;
end
end

function selected = local_selected_mask(selected_effects, Pp_sign, threshold, p, d)

if isempty(selected_effects)
    selected = Pp_sign >= threshold;
elseif islogical(selected_effects)
    selected = selected_effects;
else
    selected = false(p, d);
    selected(selected_effects) = true;
end

if ~isequal(size(selected), [p d])
    error('opts.selected_effects must be a p x d logical mask or linear indices into beta/Pp_sign.');
end

end

function tf = local_has_method(methods, name)
tf = any(strcmpi(methods, name));
end

function sgn = local_effect_sign(beta, Pp_pos)
sgn = sign(beta);
zero = sgn == 0;
sgn(zero) = sign(Pp_pos(zero) - 0.5);
sgn(sgn == 0) = 1;
end

function A = local_mask_nan(A, mask)
A(~mask) = NaN;
end

function A = local_mask_false(A, mask)
A(~mask) = false;
end

function nm = local_name(names, idx, prefix)
if isempty(names)
    nm = sprintf('%s%d', prefix, idx);
elseif iscell(names)
    nm = names{idx};
else
    nm = char(names(idx));
end
end
