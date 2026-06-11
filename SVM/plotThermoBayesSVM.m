function plotThermoBayesSVM(BSV)
% plotThermoBayesSVM
% Decision boundary and posterior probability map for 2D thermo Bayesian SVM.

    X = BSV.X;
    y = BSV.y;

    if size(X,2) ~= 2
        error('plotThermoBayesSVM currently supports only 2D X.');
    end

    % ---------------- Grid ----------------
    pad = 0.8;

    x1range = linspace(min(X(:,1))-pad, max(X(:,1))+pad, 160);
    x2range = linspace(min(X(:,2))-pad, max(X(:,2))+pad, 160);

    [xx,yy] = meshgrid(x1range, x2range);
    Xgrid = [xx(:), yy(:)];

    scoreGrid = BSV.score(Xgrid);
    probGrid = BSV.prob(Xgrid);
    scoreGrid = reshape(scoreGrid, size(xx));
    probGrid = reshape(probGrid, size(xx));

    margins = BSV.margin(X,y);

    % ---------------- Plot ----------------
    figure('Color','w','Position',[140 140 1250 520]);

    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    % --- Decision function and margins ---
    nexttile;
    hold on;

    contourf(xx, yy, scoreGrid, 32, 'LineColor','none');
    colorbar;
    contour(xx, yy, scoreGrid, [0 0], 'k', 'LineWidth', 2.5);
    contour(xx, yy, scoreGrid, [-1 -1], 'k--', 'LineWidth', 1.5);
    contour(xx, yy, scoreGrid, [1 1], 'k--', 'LineWidth', 1.5);

    scatter(X(y==-1,1), X(y==-1,2), 55, 'ko', 'filled', ...
        'MarkerFaceAlpha',0.85, 'DisplayName','Class -1');
    scatter(X(y==1,1), X(y==1,2), 55, 'wo', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, ...
        'MarkerFaceAlpha',0.85, 'DisplayName','Class +1');

    bad = margins < 1;
    scatter(X(bad,1), X(bad,2), 90, 'ro', 'LineWidth',1.8, ...
        'DisplayName','Inside margin');

    title('Thermo Bayesian SVM: decision function');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('Location','best');
    grid on; box on;
    axis tight;

    % --- Posterior class probability ---
    nexttile;
    hold on;

    contourf(xx, yy, probGrid, 32, 'LineColor','none');
    colorbar;
    contour(xx, yy, probGrid, [0.5 0.5], 'k', 'LineWidth', 2.5);

    scatter(X(y==-1,1), X(y==-1,2), 55, 'ko', 'filled', ...
        'MarkerFaceAlpha',0.85);
    scatter(X(y==1,1), X(y==1,2), 55, 'wo', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, ...
        'MarkerFaceAlpha',0.85);

    title('Posterior P(class = +1)');
    xlabel('Feature 1');
    ylabel('Feature 2');
    grid on; box on;
    axis tight;

    set(findall(gcf,'-property','FontSize'),'FontSize',14);
end