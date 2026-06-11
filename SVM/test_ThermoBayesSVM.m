function test_ThermoBayesSVM(dataset, featureMode)
% test_ThermoBayesSVM
% Test-bed for Bayesian soft-margin SVM using thermoVL.
%
% Usage:
%   test_ThermoBayesSVM
%   test_ThermoBayesSVM('moons')
%   test_ThermoBayesSVM('moons','linear')
%   test_ThermoBayesSVM('moons','quadratic')
%   test_ThermoBayesSVM('moons','rbf')
%
% Datasets:
%   diagonal_blobs
%   rotated_blobs
%   moons
%   circles
%   xor
%   outlier_blobs
%
% Feature modes:
%   linear
%   quadratic
%   rbf

    close all; clc;
    rng(42);

    if nargin < 1 || isempty(dataset)
        dataset = 'moons';
    end

    if nargin < 2 || isempty(featureMode)
        featureMode = 'rbf';
    end

    % ---------------- Generate synthetic raw 2D data ----------------
    N = 220;

    switch lower(dataset)
        case 'diagonal_blobs'
            [Xraw,y] = make_diagonal_blobs(N);

        case 'rotated_blobs'
            [Xraw,y] = make_rotated_blobs(N);

        case 'moons'
            [Xraw,y] = make_moons(N);

        case 'circles'
            [Xraw,y] = make_circles(N);

        case 'xor'
            [Xraw,y] = make_xor(N);

        case 'outlier_blobs'
            [Xraw,y] = make_outlier_blobs(N);

        otherwise
            error('Unknown dataset: %s', dataset);
    end

    % ---------------- Feature transform ----------------
    RFF = [];

    switch lower(featureMode)

        case 'linear'
            Xfit = Xraw;
            featureModeLabel = 'raw linear features';
            raw2fit = @(X) X;

        case 'quadratic'
            Xfit = quadratic_features(Xraw);
            featureModeLabel = 'quadratic features';
            raw2fit = @(X) quadratic_features(X);

        case 'rbf'
            nRFF = 80;
            gamma = 1.5;

            [Xfit, RFF] = rbf_random_features(Xraw, nRFF, gamma);

            featureModeLabel = sprintf('RBF random features: D=%d, gamma=%.2f', ...
                nRFF, gamma);

            raw2fit = @(X) rbf_random_features(X, RFF.nFeatures, RFF.gamma, RFF);

        otherwise
            error('Unknown featureMode: %s', featureMode);
    end

    % ---------------- Fit thermo Bayesian SVM ----------------
    opts = struct();

    opts.maxIter = 128;
    opts.tol = 1e-5;
    opts.plots = 0;
    opts.varpercthresh = 0.01;

    opts.C = 3;
    opts.lambda_w = 1.0;
    opts.lambda_b = 1e-3;
    opts.softplus_k = 12;

    opts.nPosteriorSamples = 1000;
    opts.prob_temperature = 1.5;

    % Slightly more regularisation for high-dimensional RBF features
    if strcmpi(featureMode,'rbf')
        opts.lambda_w = 1.5;
    end

    BSV = fitThermoBayesSVM(Xfit, y, opts);

    % ---------------- Attach raw-space plotting/evaluation metadata ----------------
    BSV.Xraw = Xraw;
    BSV.Xfit = Xfit;
    BSV.raw2fit = raw2fit;
    BSV.featureMode = featureModeLabel;
    BSV.featureModeName = featureMode;
    BSV.dataset = dataset;

    if ~isempty(RFF)
        BSV.RFF = RFF;
    end

    % Raw-space helper functions.
    BSV.score_raw = @(X) BSV.score_fit(BSV.raw2fit(X));
    BSV.predict_raw = @(X) BSV.predict_fit(BSV.raw2fit(X));
    BSV.margin_raw = @(X,ynew) BSV.margin_fit(BSV.raw2fit(X), ynew);
    BSV.prob_raw = @(X) BSV.prob_fit(BSV.raw2fit(X));

    % ---------------- Basic evaluation ----------------
    yhat = BSV.predict_fit(Xfit);
    yhat(yhat == 0) = 1;

    acc = mean(yhat == y);
    margins = BSV.margin_fit(Xfit, y);

    fprintf('\nThermo Bayesian SVM demo\n');
    fprintf('------------------------\n');
    fprintf('Dataset: %s\n', dataset);
    fprintf('Feature mode: %s\n', featureModeLabel);
    fprintf('N: %d\n', numel(y));
    fprintf('Fitted feature dimension: %d\n', size(Xfit,2));
    fprintf('Accuracy: %.2f%%\n', 100*acc);
    fprintf('Median margin: %.3f\n', median(margins));
    fprintf('Fraction inside margin: %.2f%%\n', 100*mean(margins < 1 & margins > 0));
    fprintf('Fraction misclassified: %.2f%%\n', 100*mean(margins <= 0));
    fprintf('Final ELBO: %.3f\n', BSV.logL);
    fprintf('Iterations: %d\n\n', BSV.iter);

    % ---------------- Plots ----------------
    plotThermoBayesSVM2D(BSV);
    plotThermoBayesSVMDiagnostics(BSV);
end


% ========================================================================
% Dataset generators
% ========================================================================

function [X,y] = make_diagonal_blobs(N)
    n1 = floor(N/2);
    n2 = N - n1;

    X1 = mvnrnd([-1.2 -0.8], [0.55 0.25; 0.25 0.65], n1);
    X2 = mvnrnd([ 1.1  0.9], [0.65 -0.20; -0.20 0.55], n2);

    X = [X1; X2];
    y = [-ones(n1,1); ones(n2,1)];

    y = flip_some_labels(y, 0.05);
end


function [X,y] = make_rotated_blobs(N)
    n1 = floor(N/2);
    n2 = N - n1;

    X1 = mvnrnd([-1.1  0.7], [0.75 -0.45; -0.45 0.75], n1);
    X2 = mvnrnd([ 1.1 -0.7], [0.75 -0.45; -0.45 0.75], n2);

    X = [X1; X2];
    y = [-ones(n1,1); ones(n2,1)];

    theta = pi/7;
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    X = X * R';

    y = flip_some_labels(y, 0.05);
end


function [X,y] = make_moons(N)
    n1 = floor(N/2);
    n2 = N - n1;

    t1 = linspace(0, pi, n1)';
    t2 = linspace(0, pi, n2)';

    X1 = [cos(t1), sin(t1)];
    X2 = [1 - cos(t2), 0.45 - sin(t2)];

    noise = 0.12;
    X1 = X1 + noise*randn(size(X1));
    X2 = X2 + noise*randn(size(X2));

    X = [X1; X2];
    y = [-ones(n1,1); ones(n2,1)];

    X = X - mean(X,1);

    y = flip_some_labels(y, 0.03);
end


function [X,y] = make_circles(N)
    n1 = floor(N/2);
    n2 = N - n1;

    t1 = 2*pi*rand(n1,1);
    t2 = 2*pi*rand(n2,1);

    r1 = 0.65 + 0.08*randn(n1,1);
    r2 = 1.55 + 0.12*randn(n2,1);

    X1 = [r1.*cos(t1), r1.*sin(t1)];
    X2 = [r2.*cos(t2), r2.*sin(t2)];

    X = [X1; X2];
    y = [-ones(n1,1); ones(n2,1)];

    y = flip_some_labels(y, 0.03);
end


function [X,y] = make_xor(N)
    X = 2.4 * rand(N,2) - 1.2;

    y = ones(N,1);
    y(X(:,1).*X(:,2) < 0) = -1;

    X = X + 0.08*randn(size(X));

    y = flip_some_labels(y, 0.04);
end


function [X,y] = make_outlier_blobs(N)
    n1 = floor(N/2);
    n2 = N - n1;

    X1 = mvnrnd([-1.2 0.0], [0.45 0.05; 0.05 0.45], n1);
    X2 = mvnrnd([ 1.2 0.0], [0.45 -0.05; -0.05 0.45], n2);

    X = [X1; X2];
    y = [-ones(n1,1); ones(n2,1)];

    % Add nasty outliers
    nOut = round(0.08*N);
    idx = randperm(N, nOut);

    X(idx,:) = X(idx,:) + 4.0*randn(nOut,2);

    y = flip_some_labels(y, 0.05);
end


function y = flip_some_labels(y, frac)
    N = numel(y);
    nFlip = round(frac*N);

    if nFlip > 0
        idx = randperm(N, nFlip);
        y(idx) = -y(idx);
    end
end