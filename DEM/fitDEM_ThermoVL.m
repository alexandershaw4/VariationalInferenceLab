function [POST, TRACE] = fitDEM_ThermoVL(y, f, g, x0, theta0, opts)
% fitDEM_ThermoVL  DEM-style generalised-coordinate trajectory smoother.
%
% Model: x' = f(x,theta) + w; y = g(x,theta) + z.
% D: sparse Gauss-Newton smoothing of the WHOLE state trajectory.
% E: joint Gauss-Newton parameter/state update (avoids slow EM zig-zagging).
% M: scalar Gamma precision updates for the weighted residuals.
%
% Generalised states are derivatives of ONE trajectory, rather than freely
% fitted vectors at each sample. Motion errors stop at order p-1: the missing
% derivative of order p+1 is not assumed to be zero.
%
% This is a finite-difference MAP/EM approximation in generalised coordinates,
% not the full SPM DEM variational filter. Fa is a negative penalised residual
% energy, not a calibrated ELBO or model evidence. Covariance corrections in
% the M-step, posterior state covariances and thermodynamic annealing are not
% implemented. The historical public function name is retained.
%
% Inputs: y [T,ny]; f/g @(x,theta) returning nx/ny values;
%         x0 [nx,1]; theta0 [np,1]; optional opts fields:
%   dt=1, p=2 (0:2), py=0 (0:p), Dx_iters=2, outer_iters=8,
%   gamma_y=1, gamma_x=1, lambda_y0=1/var(y(:)), lambda_x0=10,
%   pE_theta=theta0, pC_theta=10*eye(np), pE_x0, pC_x0,
%   theta_step=1, max_theta_step=2 (prior SD units),
%   varpercthresh=.01, max_rank_theta=np,
%   burnin_fix_ly=3, burnin_fix_lx=burnin_fix_ly,
%   lambda_x_target=10, lambda_x_ramp=1.8, plot=false, verbose=plot.
%
% POST.theta_V*POST.theta_V' + POST.theta_D approximates the conditional
% parameter PRECISION, not covariance. POST.theta_C is the full inverse of
% the final conditional Gauss-Newton precision (no optimiser damping).
% TRACE.dstep_cost/estep_cost give [before; after] costs at fixed precisions.

if nargin < 6, opts = struct; end
validateattributes(y, {'numeric'}, {'2d','real','finite','nonempty'}, mfilename, 'y');
validateattributes(x0, {'numeric'}, {'vector','real','finite','nonempty'}, mfilename, 'x0');
validateattributes(theta0, {'numeric'}, {'vector','real','finite','nonempty'}, mfilename, 'theta0');
y = double(y); x0 = double(x0(:)); theta = double(theta0(:));
[T,ny] = size(y); nx = numel(x0); np = numel(theta);
dt = getd(opts,'dt',1);
p = getd(opts,'p',2); py = getd(opts,'py',0);
Dx_it = getd(opts,'Dx_iters',2); outerIt = getd(opts,'outer_iters',8);
positive(dt,'dt'); integer_range(p,0,2,'p'); integer_range(py,0,p,'py');
integer_range(Dx_it,1,Inf,'Dx_iters'); integer_range(outerIt,1,Inf,'outer_iters');
if T < p+1
    error('fitDEM_ThermoVL:TooFewSamples', 'GC order p requires at least p+1 samples.');
end
model_value(f,x0,theta,nx); model_value(g,x0,theta,ny);
ngc = (p+1)*nx;

pE_theta = getd(opts,'pE_theta',theta); pE_theta = pE_theta(:);
pC_theta = getd(opts,'pC_theta',10*eye(np));
if numel(pE_theta) ~= np || any(~isfinite(pE_theta))
    error('fitDEM_ThermoVL:PriorSize', 'pE_theta must contain np finite values.');
end
Htheta = prior_precision(pC_theta,np,'pC_theta');

% Centre the default initial GC prior on x0 and its model flow.
seed = zeros(nx,p+1); seed(:,1) = x0;
if p >= 1, seed(:,2) = model_value(f,x0,theta,nx); end
if p >= 2, seed(:,3) = jacobian_fd(@(x) f(x,theta),x0)*seed(:,2); end
mu0 = getd(opts,'pE_x0',seed(:)); mu0 = mu0(:);
if numel(mu0) ~= ngc || any(~isfinite(mu0))
    error('fitDEM_ThermoVL:PriorSize', 'pE_x0 must contain (p+1)*nx finite values.');
end
C0 = getd(opts,'pC_x0',10*eye(ngc));
H0 = prior_precision(C0,ngc,'pC_x0'); L0 = chol(H0);

ly = getd(opts,'lambda_y0',1/max(var(y(:)),1e-6));
lx = getd(opts,'lambda_x0',10);
positive(ly,'lambda_y0'); positive(lx,'lambda_x0');
gamY = getd(opts,'gamma_y',1); gamX = getd(opts,'gamma_x',1);
positive(gamY,'gamma_y'); positive(gamX,'gamma_x');
% Downweight derivatives, retaining unit weight on the actual flow error.
wy = (gamY*dt).^(0:py);
wx = (gamX*dt).^(0:p-1);
if any(~isfinite([wy wx]))
    error('fitDEM_ThermoVL:Weights', 'Generalised-coordinate weights overflowed.');
end
alpha_theta = getd(opts,'theta_step',1);
max_step = getd(opts,'max_theta_step',2);
positive(alpha_theta,'theta_step'); positive(max_step,'max_theta_step');
burnY = getd(opts,'burnin_fix_ly',3); burnX = getd(opts,'burnin_fix_lx',burnY);
integer_range(burnY,0,Inf,'burnin_fix_ly'); integer_range(burnX,0,Inf,'burnin_fix_lx');
lx_target = getd(opts,'lambda_x_target',10); lx_ramp = getd(opts,'lambda_x_ramp',1.8);
positive(lx_target,'lambda_x_target'); positive(lx_ramp,'lambda_x_ramp');
varthr = getd(opts,'varpercthresh',.01); kmax = getd(opts,'max_rank_theta',np);
validateattributes(varthr,{'numeric'},{'scalar','real','finite','>=',0,'<=',1});
integer_range(kmax,1,Inf,'max_rank_theta');
doplot = getd(opts,'plot',false); verbose = getd(opts,'verbose',doplot);

% Map v=[x_1; ...; x_T] to all GC orders with sparse derivative operators.
D = derivative_operators(T,p,dt);
G = sparse(T*ngc,T*nx);
for k = 0:p
    rows = reshape(bsxfun(@plus,(k*nx+(1:nx)).',(0:T-1)*ngc),[],1);
    G(rows,:) = kron(D{k+1},speye(nx));
end
Ygc = zeros(T,(py+1)*ny);
for k = 0:py, Ygc(:,k*ny+(1:ny)) = D{k+1}*y; end
nr = (py+1)*ny + p*nx;
[ir,jc] = ndgrid(1:nr,1:ngc);
blocksize = nr*ngc;
ii = repmat(ir(:),T,1) + kron((0:T-1).'*nr,ones(blocksize,1));
jj = repmat(jc(:),T,1) + kron((0:T-1).'*ngc,ones(blocksize,1));

% Seed observable directions once, then optimise one coupled trajectory.
X = repmat(x0.',T,1);
gx = jacobian_fd(@(x) g(x,theta),x0);
for t = 1:T
    dx = gx.'*((gx*gx.'+1e-3*eye(ny))\(y(t,:).'-model_value(g,x0,theta,ny)));
    if norm(dx) > 2, dx = dx*(2/norm(dx)); end
    X(t,:) = (x0+dx).';
end
v = reshape(X.',[],1);
TRACE.Fa = zeros(1,outerIt); TRACE.sumsq_y = zeros(1,outerIt);
TRACE.sumsq_x = zeros(1,outerIt); TRACE.theta_hist = zeros(np,outerIt);
TRACE.lambda_hist = zeros(2,outerIt); TRACE.step_accepted = zeros(2,outerIt);
TRACE.dstep_cost = zeros(2,outerIt); TRACE.estep_cost = zeros(2,outerIt);
fh = [];

for it = 1:outerIt
    % D-step: samples cannot assign derivatives independent of the trajectory.
    statefun = @(z) state_residual(z,G,Ygc,theta,f,g,wy,wx,nx,ny,p,py,ly,lx,L0,mu0,ii,jj);
    cost_x = @(z) squared_cost(statefun,z);
    TRACE.dstep_cost(1,it) = cost_x(v);
    for k = 1:Dx_it
        [r,J] = statefun(v);
        dv = solve_posdef(J.'*J,-J.'*r);
        [v,accepted] = backtrack(cost_x,v,dv,1);
        TRACE.step_accepted(1,it) = TRACE.step_accepted(1,it)+accepted;
        if ~accepted, break; end
    end
    TRACE.dstep_cost(2,it) = cost_x(v);
    Xgc = reshape(G*v,ngc,T).';

    % E-step: account for the change in the hidden trajectory when theta
    % changes. Solving the joint GN system is equivalent to eliminating the
    % state block with a Schur complement; independent updates can stall.
    [rt,Jt] = parameter_residual(theta,Xgc,Ygc,f,g,wy,wx,nx,ny,p,py,ly,lx);
    [rx,Jx] = statefun(v);
    Jtheta = [sparse(Jt);sparse(ngc,np)];
    cross = Jx.'*Jtheta;
    H = [Jx.'*Jx, cross;cross.',sparse(Jt.'*Jt+Htheta)];
    b = [-Jx.'*rx;-Jt.'*rt-Htheta*(theta-pE_theta)];
    dz = solve_posdef(H,b);
    dtheta = dz(numel(v)+(1:np));
    stepnorm = norm(dtheta./sqrt(diag(pC_theta)));
    if stepnorm > max_step, dz = dz*(max_step/stepnorm); end
    cost_joint = @(z) joint_cost(z,numel(v),G,Ygc,f,g,wy,wx,nx,ny,p,py,ly,lx,L0,mu0,ii,jj,pE_theta,Htheta);
    z = [v;theta]; TRACE.estep_cost(1,it) = cost_joint(z);
    [z,accepted] = backtrack(cost_joint,z,dz,alpha_theta);
    TRACE.estep_cost(2,it) = cost_joint(z);
    v = z(1:numel(v)); theta = z(numel(v)+(1:np));
    Xgc = reshape(G*v,ngc,T).';
    TRACE.step_accepted(2,it) = accepted;

    % M-step: use exactly the weighted errors counted in the D/E steps.
    [ss_y,ss_x] = residual_sums(Xgc,Ygc,theta,f,g,wy,wx,nx,ny,p,py);
    dim_y = T*(py+1)*ny; dim_x = T*p*nx;
    if it > burnY, ly = min(1e6,max(1e-6,(dim_y+.01)/(ss_y+.01))); end
    if it > burnX && dim_x > 0
        lx = min(1e6,max(1e-6,(dim_x+.01)/(ss_x+.01)));
        lx = min(lx_target,lx*lx_ramp);
    end
    d0 = Xgc(1,:).'-mu0; dth = theta-pE_theta;
    TRACE.Fa(it) = -.5*(ly*ss_y+lx*ss_x+dth.'*Htheta*dth+d0.'*H0*d0);
    TRACE.sumsq_y(it) = ss_y; TRACE.sumsq_x(it) = ss_x;
    TRACE.theta_hist(:,it) = theta; TRACE.lambda_hist(:,it) = [ly;lx];
    if verbose
        fprintf('Iter %d | energy %.5g | ly %.4g | lx %.4g | ||dtheta|| %.4g\n', ...
            it,-TRACE.Fa(it),ly,lx,norm(dtheta));
    end
    if doplot, fh = plot_progress(fh,y,Xgc,theta,g,dt,TRACE,it,nx); end
end

% Curvature at the returned parameters/precisions, excluding optimiser damping.
[~,Jt] = parameter_residual(theta,Xgc,Ygc,f,g,wy,wx,nx,ny,p,py,ly,lx);
H = Jt.'*Jt+Htheta; H = (H+H.')/2;
[U,S] = svd(H,'econ'); lam = diag(S);
k = min(kmax,max(1,sum(lam >= varthr*lam(1))));
V = U(:,1:k)*diag(sqrt(lam(1:k)));
POST.xgc_path = Xgc; POST.x_path = Xgc(:,1:nx);
POST.theta_mean = theta; POST.theta_V = V;
POST.theta_D = diag(max(1e-12,diag(H)-sum(V.^2,2)));
L = chol(H,'lower'); POST.theta_C = L.'\(L\eye(np));
POST.lambda_y = ly; POST.lambda_x = lx;
end

function [r,J] = state_residual(v,G,Y,theta,f,g,wy,wx,nx,ny,p,py,ly,lx,L0,mu0,ii,jj)
T = size(Y,1); ngc = (p+1)*nx; nr = (py+1)*ny+p*nx;
X = reshape(G*v,ngc,T).';
rdata = zeros(nr,T);
if nargout > 1, values = zeros(T*nr*ngc,1); end
for t = 1:T
    fun = @(z) residual_gc(z,Y(t,:).',theta,f,g,wy,wx,nx,ny,p,py,ly,lx);
    xt = X(t,:).'; rdata(:,t) = fun(xt);
    if nargout > 1
        Jt = jacobian_fd(fun,xt);
        values((t-1)*nr*ngc+(1:nr*ngc)) = Jt(:);
    end
end
r = [rdata(:); L0*(X(1,:).'-mu0)];
if nargout > 1
    J = [sparse(ii,jj,values,T*nr,T*ngc)*G; L0*G(1:ngc,:)];
end
end

function [r,J] = parameter_residual(theta,X,Y,f,g,wy,wx,nx,ny,p,py,ly,lx)
T = size(X,1); nr = (py+1)*ny+p*nx;
r = zeros(T*nr,1);
if nargout > 1, J = zeros(T*nr,numel(theta)); end
for t = 1:T
    fun = @(th) residual_gc(X(t,:).',Y(t,:).',th,f,g,wy,wx,nx,ny,p,py,ly,lx);
    rows = (t-1)*nr+(1:nr); r(rows) = fun(theta);
    if nargout > 1, J(rows,:) = jacobian_fd(fun,theta); end
end
end

function [r,ey,ex] = residual_gc(xgc,ygc,theta,f,g,wy,wx,nx,ny,p,py,ly,lx)
X = reshape(xgc,nx,p+1); x = X(:,1);
Yhat = zeros(ny,py+1); Yhat(:,1) = model_value(g,x,theta,ny);
if py >= 1
    Jg = jacobian_fd(@(z) g(z,theta),x);
    Yhat(:,2) = Jg*X(:,2);
end
if py >= 2
    Yhat(:,3) = directional_second(@(z) g(z,theta),x,X(:,2))+Jg*X(:,3);
end
ey = reshape(bsxfun(@times,reshape(ygc,ny,py+1)-Yhat,wy),[],1);
F = zeros(nx,p);
if p >= 1, F(:,1) = model_value(f,x,theta,nx); end
if p >= 2, F(:,2) = jacobian_fd(@(z) f(z,theta),x)*X(:,2); end
ex = reshape(bsxfun(@times,X(:,2:end)-F,wx),[],1);
r = [sqrt(ly)*ey;sqrt(lx)*ex];
end

function [ssy,ssx] = residual_sums(X,Y,theta,f,g,wy,wx,nx,ny,p,py)
ssy = 0; ssx = 0;
for t = 1:size(X,1)
    [~,ey,ex] = residual_gc(X(t,:).',Y(t,:).',theta,f,g,wy,wx,nx,ny,p,py,1,1);
    ssy = ssy+ey.'*ey; ssx = ssx+ex.'*ex;
end
end

function D = derivative_operators(T,p,dt)
D = {speye(T)};
if p < 1, return; end
if T == 2
    D{2} = sparse([-1 1;-1 1])/dt;
else
    D{2} = spdiags([-ones(T,1),zeros(T,1),ones(T,1)],-1:1,T,T)/(2*dt);
    D{2}(1,1:3) = [-3 4 -1]/(2*dt);
    D{2}(T,T-2:T) = [1 -4 3]/(2*dt);
end
if p < 2, return; end
D{3} = spdiags([ones(T,1),-2*ones(T,1),ones(T,1)],-1:1,T,T)/dt^2;
if T == 3
    D{3}(1,:) = [1 -2 1]/dt^2; D{3}(T,:) = [1 -2 1]/dt^2;
else
    D{3}(1,1:4) = [2 -5 4 -1]/dt^2;
    D{3}(T,T-3:T) = [-1 4 -5 2]/dt^2;
end
end

function y = model_value(fun,x,theta,n)
y = fun(x,theta); y = y(:);
if numel(y) ~= n
    error('fitDEM_ThermoVL:ModelSize','f must return nx values and g ny values; y must be T-by-ny.');
end
if ~isreal(y) || any(~isfinite(y))
    error('fitDEM_ThermoVL:NonFiniteModel','The model returned complex or non-finite values.');
end
end

function J = jacobian_fd(fun,x)
y = fun(x); J = zeros(numel(y),numel(x));
for k = 1:numel(x)
    h = 1e-5*max(1,abs(x(k))); dx = zeros(size(x)); dx(k) = h;
    fp = fun(x+dx); fm = fun(x-dx);
    J(:,k) = (fp(:)-fm(:))/(2*h);
end
if ~isreal(J) || any(~isfinite(J(:)))
    error('fitDEM_ThermoVL:NonFiniteModel','The model returned a non-finite finite-difference Jacobian.');
end
end

function d2 = directional_second(fun,x,direction)
scale = max(1,norm(direction)); h = 1e-3*max(1,norm(x)); u = direction/scale;
fp = fun(x+h*u); fm = fun(x-h*u); f0 = fun(x);
d2 = (fp(:)-2*f0(:)+fm(:))*(scale/h)^2;
end

function H = prior_precision(C,n,name)
if ~isequal(size(C),[n n]) || ~isreal(C) || any(~isfinite(C(:))) || ...
        norm(C-C.','fro') > 1e-10*max(1,norm(C,'fro'))
    error('fitDEM_ThermoVL:PriorCovariance','%s must be finite, symmetric and n-by-n.',name);
end
[L,flag] = chol((C+C.')/2,'lower');
if flag, error('fitDEM_ThermoVL:PriorCovariance','%s must be positive definite.',name); end
H = L.'\(L\eye(n));
end

function dx = solve_posdef(H,b)
if any(~isfinite(nonzeros(H))) || any(~isfinite(b))
    error('fitDEM_ThermoVL:NonFiniteSystem','Cannot solve a non-finite update system.');
end
H = (H+H.')/2; n = size(H,1);
% Dimensionless LM damping, without scaling that grows with T/dt^4.
d = max(1e-12,full(diag(H))); lam = 1e-6;
for k = 1:12
    [L,flag] = chol(H+spdiags(lam*d+1e-12,0,n,n),'lower');
    if flag == 0, dx = L.'\(L\b); return; end
    lam = lam*10;
end
error('fitDEM_ThermoVL:SingularSystem','The damped update system could not be factorised.');
end

function c = squared_cost(fun,x)
r = fun(x); c = .5*(r.'*r);
if ~isreal(c) || ~isfinite(c), c = Inf; end
end

function c = joint_cost(z,nv,G,Y,f,g,wy,wx,nx,ny,p,py,ly,lx,L0,mu0,ii,jj,mu,H)
theta = z(nv+1:end);
r = state_residual(z(1:nv),G,Y,theta,f,g,wy,wx,nx,ny,p,py,ly,lx,L0,mu0,ii,jj);
d = theta-mu; c = .5*(r.'*r+d.'*H*d);
if ~isreal(c) || ~isfinite(c), c = Inf; end
end

function [xnew,accepted] = backtrack(cost,x,dx,alpha)
c0 = cost(x);
if ~isfinite(c0), error('fitDEM_ThermoVL:NonFiniteCost','Current model state has non-finite cost.'); end
xnew = x; accepted = false;
if any(~isfinite(dx)) || norm(dx) <= 1e-10*max(1,norm(x)), return; end
for k = 1:20
    try
        c = cost(x+alpha*dx);
    catch err
        if strcmp(err.identifier,'fitDEM_ThermoVL:NonFiniteModel'), c = Inf; else, rethrow(err); end
    end
    if isfinite(c) && isreal(c) && c < c0
        xnew = x+alpha*dx; accepted = true; return;
    end
    alpha = alpha/2;
end
end

function fh = plot_progress(fh,y,X,theta,g,dt,R,it,nx)
if isempty(fh), fh = figure('Name','DEM-style smoother','Color','w'); end
figure(fh); clf(fh);
T = size(y,1); pred = zeros(T,size(y,2));
for t = 1:T, yt = g(X(t,1:nx).',theta); pred(t,:) = yt(:).'; end
subplot(2,2,1); plot((0:T-1)*dt,[y(:,1),pred(:,1)]); legend('Observed','Predicted'); xlabel('Time');
subplot(2,2,2); plot(1:it,-R.Fa(1:it),'-o'); title('Penalised residual energy');
subplot(2,2,3); plot(1:it,R.lambda_hist(:,1:it).','-o'); title('Precisions'); legend('lambda y','lambda x');
subplot(2,2,4); plot(1:it,R.theta_hist(:,1:it).','-o'); title('Parameter means'); drawnow;
end

function positive(x,name)
validateattributes(x,{'numeric'},{'scalar','real','finite','positive'},mfilename,['opts.' name]);
end

function integer_range(x,lo,hi,name)
validateattributes(x,{'numeric'},{'scalar','real','finite','integer','>=',lo,'<=',hi},mfilename,['opts.' name]);
end

function value = getd(s,name,default)
if isfield(s,name) && ~isempty(s.(name)), value = s.(name); else, value = default; end
end
