function RESULT = demo_DEM_WilsonCowan(doplot, settings)
% Recover inhibitory coupling and timescale from excitatory activity alone.
% This uses a reduced two-population sigmoid Wilson-Cowan rate model.
if nargin < 1, doplot = true; end
if nargin < 2, settings = struct; end
here = fileparts(mfilename('fullpath')); addpath(here,fileparts(here));
RESULT = dem_run_demo('wilson_cowan',doplot,settings);
end
