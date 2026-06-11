function Phi = quadratic_features(X)
% quadratic_features
% Simple polynomial expansion for 2D classification.
%
% Input:
%   X: N x 2
%
% Output:
%   Phi: N x 5 = [x1, x2, x1^2, x2^2, x1*x2]

    if size(X,2) ~= 2
        error('quadratic_features expects X to be N x 2.');
    end

    x1 = X(:,1);
    x2 = X(:,2);

    Phi = [ ...
        x1, ...
        x2, ...
        x1.^2, ...
        x2.^2, ...
        x1.*x2 ...
    ];
end