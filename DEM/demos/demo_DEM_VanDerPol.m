function RESULT = demo_DEM_VanDerPol(doplot, settings)
% Recover nonlinear damping and frequency from position-only observations.
% RESULT = demo_DEM_VanDerPol(false,struct('seed',11,'noise_fraction',.1));
if nargin < 1, doplot = true; end
if nargin < 2, settings = struct; end
here = fileparts(mfilename('fullpath')); addpath(here,fileparts(here));
RESULT = dem_run_demo('vanderpol',doplot,settings);
end
