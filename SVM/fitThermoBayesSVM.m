function BSV = fitThermoBayesSVM(X, y, opts)
% Bayesian soft-margin SVM using fitVariationalLaplaceThermo.
%
% X : N x P design matrix
% y : N x 1 labels, coded as -1/+1 or 0/1
%
% Returns:
%   BSV.m      posterior mean parameters [w; b]
%   BSV.V,D    posterior precision representation from thermoVL
%   BSV.logL   final ELBO
%   BSV.predict function handle
%   BSV.score   function handle

    if nargin < 3 || isempty(opts), opts = struct; end

    if ~isfield(opts,'maxIter'), opts.maxIter = 128; end
    if ~isfield(opts,'tol'), opts.tol = 1e-5; end
    if ~isfield(opts,'plots'), opts.plots = 1; end
    if ~isfield(opts,'varpercthresh'), opts.varpercthresh = 0.01; end

    % SVM-ish hyperparameters
    if ~isfield(opts,'C'), opts.C = 1; end              % margin sharpness / data weight proxy
    if ~isfield(opts,'lambda_w'), opts.lambda_w = 1; end % weight prior precision
    if ~isfield(opts,'lambda_b'), opts.lambda_b = 1e-3; end
    if ~isfield(opts,'softplus_k'), opts.softplus_k = 10; end

    y = y(:);
    if all(ismember(unique(y), [0 1]))
        y = 2*y - 1;
    end

    [N,P] = size(X);

    % Standardise predictors
    muX = mean(X,1);
    sdX = std(X,[],1);
    sdX(sdX == 0) = 1;
    Xz = (X - muX) ./ sdX;

    % Parameter vector theta = [w; b]
    m0 = zeros(P+1,1);

    % Prior covariance. Smaller variance = stronger regularisation.
    S0 = diag([ones(P,1)./opts.lambda_w; 1./opts.lambda_b]);

    % Observations are zero violations
    yobs = zeros(N,1);

    % Model returns smooth hinge violation
    f = @(theta) bayes_svm_violation(theta, Xz, y, opts.softplus_k, opts.C);

    [m,V,D,logL,iter,sigma2,allm,g_elbo] = ...
        fitVariationalLaplaceThermo(yobs, f, m0, S0, ...
            opts.maxIter, opts.tol, opts.plots, opts.varpercthresh);

    BSV = struct();
    BSV.m = m;
    BSV.V = V;
    BSV.D = D;
    BSV.logL = logL;
    BSV.iter = iter;
    BSV.sigma2 = sigma2;
    BSV.allm = allm;
    BSV.g_elbo = g_elbo;
    BSV.muX = muX;
    BSV.sdX = sdX;
    BSV.opts = opts;

    BSV.score = @(Xnew) thermo_svm_score(BSV, Xnew);
    BSV.predict = @(Xnew) sign(thermo_svm_score(BSV, Xnew));
    BSV.margin = @(Xnew, ynew) ynew(:) .* thermo_svm_score(BSV, Xnew);
end


function v = bayes_svm_violation(theta, X, y, k, C)
    P = size(X,2);
    w = theta(1:P);
    b = theta(P+1);

    score = X*w + b;
    margin = y .* score;

    % Smooth hinge: max(0, 1 - margin)
    z = k * (1 - margin);

    % Numerically stable softplus
    softplus_z = max(z,0) + log1p(exp(-abs(z)));

    v = sqrt(C) * softplus_z / k;
end


function s = thermo_svm_score(BSV, Xnew)
    Xz = (Xnew - BSV.muX) ./ BSV.sdX;
    P = size(Xz,2);
    w = BSV.m(1:P);
    b = BSV.m(P+1);
    s = Xz*w + b;
end