function BSV = fitThermoBayesSVM(X, y, opts)
% fitThermoBayesSVM
% Bayesian soft-margin SVM using fitVariationalLaplaceThermo.
%
% X : N x P predictor matrix in FITTED FEATURE SPACE
% y : N x 1 class labels, either {-1,+1} or {0,1}
%
% This routine expects X to already be the feature matrix you want to fit:
%   raw linear:     X = [x1 x2 ...]
%   quadratic:      X = quadratic_features(Xraw)
%   RBF random:     X = rbf_random_features(Xraw,...)
%
% The fitted model predicts soft margin violations:
%   violation_i = softplus(k * (1 - y_i * score_i)) / k
%
% where:
%   score_i = X_i*w + b
%
% The target is zero violation.

    if nargin < 3 || isempty(opts), opts = struct(); end

    % ---------------- Defaults ----------------
    if ~isfield(opts,'maxIter'),       opts.maxIter = 128; end
    if ~isfield(opts,'tol'),           opts.tol = 1e-5; end
    if ~isfield(opts,'plots'),         opts.plots = 1; end
    if ~isfield(opts,'varpercthresh'), opts.varpercthresh = 0.01; end

    % SVM-like controls
    if ~isfield(opts,'C'),             opts.C = 1; end
    if ~isfield(opts,'lambda_w'),      opts.lambda_w = 1; end
    if ~isfield(opts,'lambda_b'),      opts.lambda_b = 1e-3; end
    if ~isfield(opts,'softplus_k'),    opts.softplus_k = 10; end

    % Posterior prediction controls
    if ~isfield(opts,'nPosteriorSamples'), opts.nPosteriorSamples = 500; end
    if ~isfield(opts,'prob_temperature'),  opts.prob_temperature = 1.0; end

    % ---------------- Labels ----------------
    y = y(:);

    if all(ismember(unique(y), [0 1]))
        y = 2*y - 1;
    end

    if ~all(ismember(unique(y), [-1 1]))
        error('Labels y must be coded as {-1,+1} or {0,1}.');
    end

    [N,P] = size(X);

    % ---------------- Standardise fitted feature space ----------------
    muX = mean(X,1);
    sdX = std(X,[],1);
    sdX(sdX == 0) = 1;

    Xz = (X - muX) ./ sdX;

    % ---------------- Priors ----------------
    % theta = [w; b]
    m0 = zeros(P+1,1);

    prior_var_w = 1 ./ opts.lambda_w;
    prior_var_b = 1 ./ opts.lambda_b;

    S0 = diag([prior_var_w * ones(P,1); prior_var_b]);

    % ---------------- Target: zero margin violation ----------------
    yobs = zeros(N,1);

    f = @(theta) thermo_bsvm_violation(theta, Xz, y, opts.softplus_k, opts.C);

    % ---------------- Fit thermoVL ----------------
    [m,V,D,logL,iter,sigma2,allm,g_elbo] = ...
        fitVariationalLaplaceThermo(yobs, f, m0, S0, ...
        opts.maxIter, opts.tol, opts.plots, opts.varpercthresh);

    % ---------------- Package output ----------------
    BSV = struct();

    BSV.m = m;
    BSV.V = V;
    BSV.D = D;
    BSV.logL = logL;
    BSV.iter = iter;
    BSV.sigma2 = sigma2;
    BSV.allm = allm;
    BSV.g_elbo = g_elbo;

    BSV.X = X;
    BSV.Xfit = X;
    BSV.Xz = Xz;
    BSV.y = y;
    BSV.muX = muX;
    BSV.sdX = sdX;
    BSV.opts = opts;

    % Explicit fitted-feature-space functions.
    BSV.score_fit   = @(XnewFit) thermo_bsvm_score(BSV, XnewFit);
    BSV.predict_fit = @(XnewFit) sign(thermo_bsvm_score(BSV, XnewFit));
    BSV.margin_fit  = @(XnewFit,ynew) ynew(:) .* thermo_bsvm_score(BSV, XnewFit);
    BSV.prob_fit    = @(XnewFit) thermo_bsvm_posterior_prob(BSV, XnewFit, opts.nPosteriorSamples);

    % Backwards-compatible aliases. These also expect fitted feature space.
    BSV.score   = BSV.score_fit;
    BSV.predict = BSV.predict_fit;
    BSV.margin  = BSV.margin_fit;
    BSV.prob    = BSV.prob_fit;
end


function v = thermo_bsvm_violation(theta, Xz, y, k, C)
    P = size(Xz,2);

    w = theta(1:P);
    b = theta(P+1);

    score = Xz*w + b;
    margin = y .* score;

    % Smooth hinge approximation to max(0, 1 - margin)
    z = k * (1 - margin);

    % Numerically stable softplus
    softplus_z = max(z,0) + log1p(exp(-abs(z)));

    % sqrt(C) makes squared-error loss behave approximately like C * hinge^2
    v = sqrt(C) * softplus_z / k;
end


function score = thermo_bsvm_score(BSV, XnewFit)
    if size(XnewFit,2) ~= numel(BSV.muX)
        error(['Feature-space mismatch in thermo_bsvm_score: Xnew has %d columns, ' ...
               'but model expects %d columns. Transform raw data first.'], ...
               size(XnewFit,2), numel(BSV.muX));
    end

    Xz = (XnewFit - BSV.muX) ./ BSV.sdX;

    P = size(Xz,2);

    w = BSV.m(1:P);
    b = BSV.m(P+1);

    score = Xz*w + b;
end


function p = thermo_bsvm_posterior_prob(BSV, XnewFit, nSamp)
% Posterior probability P(class = +1) by sampling decision boundaries.

    if nargin < 3 || isempty(nSamp)
        nSamp = 500;
    end

    if size(XnewFit,2) ~= numel(BSV.muX)
        error(['Feature-space mismatch in thermo_bsvm_posterior_prob: Xnew has %d columns, ' ...
               'but model expects %d columns. Transform raw data first.'], ...
               size(XnewFit,2), numel(BSV.muX));
    end

    temp = BSV.opts.prob_temperature;

    Xz = (XnewFit - BSV.muX) ./ BSV.sdX;
    P = size(Xz,2);

    Happrox = BSV.V*BSV.V' + diag(BSV.D);
    Happrox = (Happrox + Happrox')/2;

    jitter = 1e-6 * max(1, max(abs(diag(Happrox))));
    Sigma = inv(Happrox + jitter*eye(size(Happrox)));

    Sigma = (Sigma + Sigma')/2;

    try
        theta_samps = mvnrnd(BSV.m(:)', Sigma, nSamp);
    catch
        [Q,L] = eig(Sigma);
        lam = max(diag(L), 1e-10);
        Sigma = Q*diag(lam)*Q';
        theta_samps = mvnrnd(BSV.m(:)', Sigma, nSamp);
    end

    scores = zeros(size(Xz,1), nSamp);

    for s = 1:nSamp
        th = theta_samps(s,:)';
        w = th(1:P);
        b = th(P+1);
        scores(:,s) = Xz*w + b;
    end

    % Softer Bayesian probability rather than hard majority vote.
    p = mean(1 ./ (1 + exp(-scores ./ temp)), 2);
end