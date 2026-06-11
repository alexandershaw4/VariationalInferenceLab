function [Z, RFF] = rbf_random_features(X, nFeatures, gamma, RFF)
% rbf_random_features
% Random Fourier feature approximation to an RBF kernel.
%
% k(x,x') = exp(-gamma * ||x - x'||^2)
%
% Inputs:
%   X          : N x P raw data
%   nFeatures : number of random Fourier features
%   gamma      : RBF kernel width parameter
%   RFF        : optional struct with fixed W,b for test/grid reuse
%
% Outputs:
%   Z   : N x nFeatures transformed data
%   RFF : struct containing W,b,gamma,nFeatures

    if nargin < 4 || isempty(RFF)

        if nargin < 2 || isempty(nFeatures)
            nFeatures = 80;
        end

        if nargin < 3 || isempty(gamma)
            gamma = 1.5;
        end

        P = size(X,2);

        RFF = struct();
        RFF.W = sqrt(2*gamma) * randn(P, nFeatures);
        RFF.b = 2*pi*rand(1, nFeatures);
        RFF.gamma = gamma;
        RFF.nFeatures = nFeatures;
    end

    Z = sqrt(2/RFF.nFeatures) * cos(X * RFF.W + RFF.b);
end