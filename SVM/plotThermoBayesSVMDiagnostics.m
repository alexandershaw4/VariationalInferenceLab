function plotThermoBayesSVMDiagnostics(BSV)
% plotThermoBayesSVMDiagnostics
% Diagnostic plots for thermo Bayesian SVM.

    if isfield(BSV,'Xfit')
        Xfit = BSV.Xfit;
    else
        Xfit = BSV.X;
    end

    y = BSV.y;

    margins = BSV.margin_fit(Xfit,y);
    scores = BSV.score_fit(Xfit);

    yhat = sign(scores);
    yhat(yhat == 0) = 1;

    acc = mean(yhat == y);

    theta = BSV.m(:);
    P = numel(theta);

    % Approx posterior covariance from precision-like output.
    Happrox = BSV.V*BSV.V' + diag(BSV.D);
    Happrox = (Happrox + Happrox')/2;

    jitter = 1e-6 * max(1,max(abs(diag(Happrox))));
    Sigma = inv(Happrox + jitter*eye(size(Happrox)));

    Sigma = (Sigma + Sigma')/2;
    se = sqrt(max(0,diag(Sigma)));

    % Avoid unreadable parameter plot if many RBF features.
    maxParamsToPlot = min(P, 40);

    figure('Color','w','Position',[160 160 1400 800]);
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    % =====================================================================
    % Margins
    % =====================================================================
    nexttile;
    histogram(margins, 28);
    hold on;
    xline(0,'r-','LineWidth',2);
    xline(1,'k--','LineWidth',2);
    hold off;

    title(sprintf('Margins | accuracy %.1f%%',100*acc));
    xlabel('y_i f(x_i)');
    ylabel('Count');
    grid on; box on;

    % =====================================================================
    % Scores by class
    % =====================================================================
    nexttile;
    hold on;

    histogram(scores(y==-1), 24, 'FaceAlpha',0.55);
    histogram(scores(y==1), 24, 'FaceAlpha',0.55);
    xline(0,'k-','LineWidth',2);

    hold off;

    title('Decision scores by class');
    xlabel('f(x)');
    ylabel('Count');
    legend({'Class -1','Class +1'},'Location','best');
    grid on; box on;

    % =====================================================================
    % Posterior parameters
    % =====================================================================
    nexttile;
    errorbar(1:maxParamsToPlot, theta(1:maxParamsToPlot), ...
        1.96*se(1:maxParamsToPlot), ...
        'ko', 'MarkerFaceColor','k', 'LineWidth',1.4);

    yline(0,'k--');

    if P > maxParamsToPlot
        title(sprintf('Posterior parameters first %d/%d', maxParamsToPlot, P));
    else
        title('Approximate posterior over parameters');
    end

    xlabel('Parameter index');
    ylabel('Posterior mean ± 95% interval');
    grid on; box on;

    % =====================================================================
    % Parameter trajectories
    % =====================================================================
    nexttile;

    if isfield(BSV,'allm') && ~isempty(BSV.allm)
        nTraj = min(size(BSV.allm,1), 20);
        plot(BSV.allm(1:nTraj,:)', 'LineWidth',1.2);

        if size(BSV.allm,1) > nTraj
            title(sprintf('Parameter trajectories first %d/%d', nTraj, size(BSV.allm,1)));
        else
            title('Parameter trajectories');
        end

        xlabel('Iteration');
        ylabel('\theta');
        grid on; box on;
    else
        text(0.5,0.5,'No trajectory available','HorizontalAlignment','center');
        axis off;
    end

    % =====================================================================
    % Heteroscedastic sigma
    % =====================================================================
    nexttile;

    if isfield(BSV,'sigma2') && ~isempty(BSV.sigma2)
        plot(sqrt(BSV.sigma2),'ko-','MarkerFaceColor','k','LineWidth',1.2);

        title('Estimated observation σ on margin violations');
        xlabel('Observation');
        ylabel('\sigma');
        grid on; box on;
    else
        text(0.5,0.5,'No sigma2 available','HorizontalAlignment','center');
        axis off;
    end

    % =====================================================================
    % Confusion matrix
    % =====================================================================
    nexttile;

    cm = zeros(2,2);
    labs = [-1 1];

    for i = 1:2
        for j = 1:2
            cm(i,j) = sum(y == labs(i) & yhat == labs(j));
        end
    end

    imagesc(cm);
    axis square;
    colorbar;

    xticks(1:2);
    yticks(1:2);

    xticklabels({'Pred -1','Pred +1'});
    yticklabels({'True -1','True +1'});

    title('Confusion matrix');

    for i = 1:2
        for j = 1:2
            text(j,i,num2str(cm(i,j)), ...
                'HorizontalAlignment','center', ...
                'FontWeight','bold', ...
                'FontSize',14);
        end
    end

    set(findall(gcf,'-property','FontSize'),'FontSize',14);
end