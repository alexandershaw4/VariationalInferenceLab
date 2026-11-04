function fh = plot_DEM_demo(R,visible)
% Replot a saved numeric RESULT without rerunning its fit.
% fh = plot_DEM_demo(RESULT,'off'); print(fh,'result.png','-dpng','-r150');
if nargin < 2, visible = 'on'; end
M = R.model; T = R.TRACE;
fh = figure('Color','w','Name',[M.label ' - hidden state recovery'], ...
    'Visible',visible,'Position',[100 100 1200 700]);
set(fh,'PaperPositionMode','auto');
blue = [0 .38 .72]; orange = [.85 .34 .07];
subplot(2,3,1);
plot(R.time,R.y,'.','Color',[.68 .68 .68]); hold on;
plot(R.time,R.y_true,'k-','LineWidth',1.1);
plot(R.time,R.yhat,'-','Color',blue,'LineWidth',1.3);
plot(R.time,R.x_free(:,1),'--','Color',orange,'LineWidth',1);
title([M.label ': observed state']); xlabel('Time (model units)'); ylabel(M.state_names{1});
small_legend({'Noisy data','Truth','Smoothed','Free simulation'});
subplot(2,3,2);
plot(R.time,R.x_true(:,2),'k-','LineWidth',1.1); hold on;
plot(R.time,R.POST.x_path(:,2),'-','Color',blue,'LineWidth',1.3);
title('Hidden state'); xlabel('Time (model units)'); ylabel(M.state_names{2});
small_legend({'Truth','Reconstructed'});
subplot(2,3,3);
plot(R.x_true(:,1),R.x_true(:,2),'k-','LineWidth',1.1); hold on;
plot(R.POST.x_path(:,1),R.POST.x_path(:,2),'-','Color',blue,'LineWidth',1.3);
xlabel(M.state_names{1}); ylabel(M.state_names{2}); title('Phase portrait');
small_legend({'Truth','Reconstructed'});
physical = [R.theta0,exp(T.theta_hist)]; its = 0:size(T.theta_hist,2);
for k = 1:2
    subplot(2,3,3+k);
    plot(its,physical(k,:),'-o','Color',blue,'LineWidth',1.2,'MarkerSize',4); hold on;
    plot(its,R.theta_true(k)*ones(size(its)),'k--','LineWidth',1);
    xlabel('Outer iteration'); ylabel(M.theta_names{k});
    title(sprintf('Parameter error %.1f%%',100*R.metrics.parameter_relative_error(k)));
    lg = small_legend({'Estimate','Truth'});
    set(lg,'Location','southeast');
end
subplot(2,3,6);
its = 1:size(T.lambda_hist,2);
semilogy(its,T.lambda_hist(1,:),'-o','Color',blue,'LineWidth',1.2,'MarkerSize',4); hold on;
semilogy(its,T.lambda_hist(2,:),'-s','Color',orange,'LineWidth',1.2,'MarkerSize',4);
semilogy(its,ones(size(its))/R.noise_sd^2,'k--');
xlabel('Outer iteration'); ylabel('Precision'); title('Precision updates');
small_legend({'Observation','Dynamics','Simulated observation'});
end

function lg = small_legend(labels)
lg = legend(labels,'Location','northeast');
set(lg,'FontSize',8);
end
