function Phi = quadratic_features(X)
% quadratic_features
% Polynomial expansion up to quadratic terms for N-dimensional data.
%
% Input:
%   X: N x D
%
% Output:
%   Phi: N x P, where
%        P = D linear terms
%          + D squared terms
%          + D*(D-1)/2 pairwise interaction terms
%
% Feature order:
%   [x1 ... xD, x1^2 ... xD^2, x1*x2, x1*x3, ..., x(D-1)*xD]

    if ~isnumeric(X) || ndims(X) ~= 2
        error('quadratic_features expects X to be a numeric N x D matrix.');
    end

    [N, D] = size(X);

    if D < 1
        error('X must have at least one column.');
    end

    % Linear terms
    linear_terms = X;

    % Squared terms
    squared_terms = X.^2;

    % Pairwise interaction terms
    n_interactions = D * (D - 1) / 2;
    interaction_terms = zeros(N, n_interactions);

    k = 1;
    for i = 1:D-1
        for j = i+1:D
            interaction_terms(:, k) = X(:, i) .* X(:, j);
            k = k + 1;
        end
    end

    % Combine
    Phi = [linear_terms, squared_terms, interaction_terms];
end