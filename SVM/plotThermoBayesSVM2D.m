function plotThermoBayesSVM2D(BSV)
% plotThermoBayesSVM2D
% Plots decision boundary in raw 2D space, even if the model was fitted
% using transformed features such as quadratic or RBF random features.

    if isfield(BSV,'Xraw')
        Xraw = BSV.Xraw;
    else
        Xraw = BSV.X;
    end

    y = BSV.y;

    if size(Xraw,2) ~= 2
        error('plotThermoBayesSVM2D requires raw 2D coordinates.');
    end

    % Compute margins in fitted feature space.
    if isfield(BSV,'raw2fit')
        Xfit = BSV.raw2fit(Xraw);
    elseif isfield(BSV,'Xfit')
        Xfit = BSV.Xfit;
    else
        Xfit = Xraw;
    end

    margins = BSV.margin_fit(Xfit, y);

    insideMargin = margins < 1 & margins > 0;
    misclassified = margins <= 0;

    % ---------------- Grid in raw 2D coordinates ----------------
    pad = 0.8;

    x1range = linspace(min(Xraw(:,1))-pad, max(Xraw(:,1))+pad, 200);
    x2range = linspace(min(Xraw(:,2))-pad, max(Xraw(:,2))+pad, 200);

    [xx,yy] = meshgrid(x1range, x2range);
    XgridRaw = [xx(:), yy(:)];

    % Score/prob in raw space via raw2fit.
    if isfield(BSV,'score_raw')
        scoreGrid = BSV.score_raw(XgridRaw);
    else
        XgridFit = BSV.raw2fit(XgridRaw);
        scoreGrid = BSV.score_fit(XgridFit);
    end

    if isfield(BSV,'prob_raw')
        probGrid = BSV.prob_raw(XgridRaw);
    else
        XgridFit = BSV.raw2fit(XgridRaw);
        probGrid = BSV.prob_fit(XgridFit);
    end

    scoreGrid = reshape(scoreGrid, size(xx));
    probGrid = reshape(probGrid, size(xx));

    % ---------------- Labels ----------------
    datasetLabel = 'dataset';
    featureLabel = 'features';

    if isfield(BSV,'dataset')
        datasetLabel = BSV.dataset;
    end

    if isfield(BSV,'featureMode')
        featureLabel = BSV.featureMode;
    end

    % ---------------- Plot ----------------
    figure('Color','w','Position',[120 120 1350 580]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    % =====================================================================
    % Decision function
    % =====================================================================
    nexttile;
    hold on;

    contourf(xx, yy, scoreGrid, 40, 'LineColor','none');
    colorbar;

    contour(xx, yy, scoreGrid, [0 0], 'k', 'LineWidth', 2.5);
    contour(xx, yy, scoreGrid, [-1 -1], 'k--', 'LineWidth', 1.5);
    contour(xx, yy, scoreGrid, [1 1], 'k--', 'LineWidth', 1.5);

    hNeg = scatter(Xraw(y==-1,1), Xraw(y==-1,2), ...
        45, 'ko', 'filled', 'MarkerFaceAlpha',0.85);

    hPos = scatter(Xraw(y==1,1), Xraw(y==1,2), ...
        45, 'wo', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, ...
        'MarkerFaceAlpha',0.85);

    hInside = scatter(Xraw(insideMargin,1), Xraw(insideMargin,2), ...
        85, 'ro', 'LineWidth',1.5);

    hWrong = scatter(Xraw(misclassified,1), Xraw(misclassified,2), ...
        120, 'rx', 'LineWidth',2.4);

    title(sprintf('Decision function: %s / %s', datasetLabel, featureLabel), ...
        'Interpreter','none');

    xlabel('Feature 1');
    ylabel('Feature 2');

    legend([hNeg hPos hInside hWrong], ...
        {'Class -1','Class +1','Inside margin','Misclassified'}, ...
        'Location','best');

    grid on; box on;
    axis tight;

    % =====================================================================
    % Posterior probability
    % =====================================================================
    nexttile;
    hold on;

    contourf(xx, yy, probGrid, 40, 'LineColor','none');
    colorbar;

    contour(xx, yy, probGrid, [0.5 0.5], 'k', 'LineWidth', 2.5);

    scatter(Xraw(y==-1,1), Xraw(y==-1,2), ...
        45, 'ko', 'filled', 'MarkerFaceAlpha',0.85);

    scatter(Xraw(y==1,1), Xraw(y==1,2), ...
        45, 'wo', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, ...
        'MarkerFaceAlpha',0.85);

    title('Posterior P(class = +1)');
    xlabel('Feature 1');
    ylabel('Feature 2');

    grid on; box on;
    axis tight;

    set(findall(gcf,'-property','FontSize'),'FontSize',14);
end