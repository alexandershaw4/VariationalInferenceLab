function [yhat, beta, X, stats] = smooth_dct_fit(y, nd, varargin)
%SMOOTH_DCT_FIT Smooth a vector using a DCT basis regression.
%
%   [yhat, beta, X, stats] = smooth_dct_fit(y, nd)
%
% Uses:
%   X = spm_dctmtx(nf, nd + 1);
%
% Inputs
%   y   : vector, e.g. parameter values across sleep windows
%   nd  : number of DCT components minus intercept
%
% Optional name-value inputs
%   'standardise' : true/false, z-score y before fitting, default false
%   'robust'      : true/false, use robustfit if available, default false
%
% Outputs
%   yhat  : fitted smooth vector, same shape as y
%   beta  : DCT regression coefficients
%   X     : DCT design matrix
%   stats : struct with R2, residuals, nd, nf

% Defaults
standardise = false;
use_robust  = false;

for i = 1:2:numel(varargin)
    switch lower(varargin{i})
        case 'standardise'
            standardise = varargin{i+1};
        case 'robust'
            use_robust = varargin{i+1};
        otherwise
            error('Unknown option: %s', varargin{i});
    end
end

% Preserve input orientation
was_row = isrow(y);
y = y(:);

nf = numel(y);

% Build DCT basis
X = spm_dctmtx(nf, nd + 1);

% Handle missing values
good = ~isnan(y) & ~isinf(y);

if sum(good) < size(X,2)
    warning('Fewer valid data points than basis functions. Reducing nd.');

    nd_new = max(0, sum(good) - 1);
    X = spm_dctmtx(nf, nd_new + 1);
    nd = nd_new;
end

% Optional standardisation
mu = 0;
sd = 1;

yf = y;

if standardise
    mu = nanmean(yf);
    sd = nanstd(yf);

    if sd == 0 || isnan(sd)
        sd = 1;
    end

    yf = (yf - mu) ./ sd;
end

% Fit only valid rows
Xg = X(good,:);
yg = yf(good);

if use_robust && exist('robustfit', 'file') == 2
    % robustfit adds an intercept by default, so turn it off
    beta = robustfit(Xg, yg, [], [], 'off');
else
    beta = Xg \ yg;
end

% Predict full trajectory
yhat = X * beta;

% Back-transform
if standardise
    yhat = yhat .* sd + mu;
end

% Residuals/statistics
res = y - yhat;

ss_res = nansum(res.^2);
ss_tot = nansum((y - nanmean(y)).^2);

if ss_tot > 0
    R2 = 1 - ss_res / ss_tot;
else
    R2 = NaN;
end

stats = struct();
stats.nf = nf;
stats.nd = nd;
stats.n_basis = size(X,2);
stats.good = good;
stats.residuals = res;
stats.R2 = R2;
stats.mu = mu;
stats.sd = sd;

% Restore row orientation
if was_row
    yhat = yhat';
    res = res';
    stats.residuals = res;
end

end