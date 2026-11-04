function X = dem_rk4(f,x0,phi,time,substeps)
% Simulate on a finer grid than the observations, independently of the
% fitter's finite-difference trajectory representation.
X = zeros(numel(time),numel(x0)); X(1,:) = x0(:).';
for t = 2:numel(time)
    h = (time(t)-time(t-1))/substeps; x = X(t-1,:).';
    for j = 1:substeps
        a = f(x,phi); b = f(x+h*a/2,phi);
        c = f(x+h*b/2,phi); d = f(x+h*c,phi);
        x = x+h*(a+2*b+2*c+d)/6;
    end
    if ~isreal(x) || any(~isfinite(x))
        error('DEM_demos:SimulationFailed','Non-finite simulation at sample %d.',t);
    end
    X(t,:) = x.';
end
end
