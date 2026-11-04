function [results, summary] = run_DEM_demos(config)
% Run the three demonstrations, or a grid of noise levels and random seeds.
%
%   results = run_DEM_demos;             % three fits and three figures
%   results = run_DEM_demos(struct('plot',false));
%   [results,summary] = run_DEM_demos(struct( ...
%       'plot',false,'noise_fractions',[.03 .10],'seeds',[11 29], ...
%       'output_dir','DEM_results'));    % 12 fits; MAT files and summary CSV
%
% No files are written unless output_dir is supplied. Set save_plots=true to
% export PNG figures as well; a working MATLAB/Octave graphics backend is
% then required, including when plot=false. settings passes common overrides
% to each individual demo. Failed recovery checks are reported, never dropped.
if nargin < 1, config = struct; end
allowed = {'models','plot','noise_fractions','seeds','output_dir','save_plots','verbose','settings'};
if ~isstruct(config) || ~isscalar(config), error('DEM_demos:Config','config must be a scalar structure.'); end
names = fieldnames(config);
for k = 1:numel(names)
    if ~any(strcmp(names{k},allowed)), error('DEM_demos:Config','Unknown config field: %s',names{k}); end
end
models = getfield_default(config,'models',{'vanderpol','fitzhugh_nagumo','wilson_cowan'});
noise = getfield_default(config,'noise_fractions',.05);
seeds = getfield_default(config,'seeds',42);
doplot = getfield_default(config,'plot',true);
outdir = getfield_default(config,'output_dir','');
save_plots = getfield_default(config,'save_plots',false);
verbose = getfield_default(config,'verbose',true);
common = getfield_default(config,'settings',struct);
validateattributes(noise,{'numeric'},{'vector','real','finite','positive','nonempty'});
validateattributes(seeds,{'numeric'},{'vector','integer','>=',0,'<=',2^32-1,'nonempty'});
if ~iscellstr(models) || isempty(models), error('DEM_demos:Config','models must be a nonempty cell array of model IDs.'); end
if ~isempty(outdir) && ~exist(outdir,'dir'), mkdir(outdir); end
if save_plots && isempty(outdir), error('DEM_demos:Config','save_plots requires an output_dir.'); end
here = fileparts(mfilename('fullpath')); addpath(here,fileparts(here));
results = cell(numel(models)*numel(noise)*numel(seeds),1);
summary = struct([]); n = 0;
for m = 1:numel(models)
    for a = 1:numel(noise)
        for b = 1:numel(seeds)
            settings = common; settings.noise_fraction = noise(a);
            settings.seed = seeds(b); settings.verbose = verbose;
            R = dem_run_demo(models{m},doplot,settings);
            n = n+1; results{n} = R;
            s.model = R.model.id; s.noise_fraction = noise(a); s.seed = seeds(b);
            s.parameter1 = R.model.theta_names{1}; s.true1 = R.theta_true(1);
            s.estimate1 = R.theta_est(1); s.error1_pct = 100*R.metrics.parameter_relative_error(1);
            s.parameter2 = R.model.theta_names{2}; s.true2 = R.theta_true(2);
            s.estimate2 = R.theta_est(2); s.error2_pct = 100*R.metrics.parameter_relative_error(2);
            s.observed_nrmse = R.metrics.state_nrmse(1); s.hidden_nrmse = R.metrics.state_nrmse(2);
            s.free_run_observed_nrmse = R.metrics.free_run_nrmse(1);
            s.noise_sd = R.noise_sd; s.noise_sd_est = R.metrics.noise_sd_est;
            s.elapsed_seconds = R.elapsed_seconds; s.recovery_ok = R.metrics.recovery_ok;
            summary(n) = s; %#ok<AGROW>
            if ~isempty(outdir)
                tag = sprintf('%s_noise%s_seed%u',models{m},strrep(sprintf('%.9g',noise(a)),'.','p'),seeds(b));
                save(fullfile(outdir,[tag '.mat']),'R','-v7');
                if save_plots
                    fh = plot_DEM_demo(R,'off');
                    print(fh,fullfile(outdir,[tag '.png']),'-dpng','-r150'); close(fh);
                end
            end
        end
    end
end
fprintf('\n%-19s %7s %6s %9s %9s %12s %8s\n', ...
    'Model','Noise %','Seed','Param1 %','Param2 %','Hidden NRMSE','Recovery');
for k = 1:numel(summary)
    s = summary(k);
    fprintf('%-19s %7.1f %6u %9.2f %9.2f %12.4f %8d\n',s.model, ...
        100*s.noise_fraction,s.seed,s.error1_pct,s.error2_pct,s.hidden_nrmse,s.recovery_ok);
end
fprintf('%d/%d runs met the stated parameter/state recovery thresholds.\n',sum([summary.recovery_ok]),numel(summary));
if ~isempty(outdir)
    write_summary(fullfile(outdir,'summary.csv'),summary);
    save(fullfile(outdir,'all_results.mat'),'results','summary','-v7');
end
end

function value = getfield_default(s,name,default)
if isfield(s,name), value = s.(name); else, value = default; end
end

function write_summary(path,S)
fid = fopen(path,'w');
if fid < 0, error('DEM_demos:Output','Cannot write %s',path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
names = fieldnames(S);
fprintf(fid,'%s\n',strjoin(names,','));
for k = 1:numel(S)
    values = cell(1,numel(names));
    for j = 1:numel(names)
        value = S(k).(names{j});
        if ischar(value), values{j} = ['"' strrep(value,'"','""') '"'];
        else, values{j} = sprintf('%.12g',value); end
    end
    fprintf(fid,'%s\n',strjoin(values,','));
end
end
