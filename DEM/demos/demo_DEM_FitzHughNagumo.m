function RESULT = demo_DEM_FitzHughNagumo(doplot, settings)
% Recover constant drive and recovery timescale from voltage alone.
% The recovery variable is hidden; a=.7 and b=.8 are fixed.
if nargin < 1, doplot = true; end
if nargin < 2, settings = struct; end
here = fileparts(mfilename('fullpath')); addpath(here,fileparts(here));
RESULT = dem_run_demo('fitzhugh_nagumo',doplot,settings);
end
