% 【檔案功能】評估：畫出 AUC 與準確率對維度的曲線（各折平均），存成圖檔。
%   PCA 維度實驗與 LDA 維度實驗共用這個函式。

function plot_dimension_curves(dims, accuracy, auc, seriesNames, numFolds, ...
    fileName, labels)
%PLOT_DIMENSION_CURVES AUC and accuracy against the dimension, saved to a file.
%   PLOT_DIMENSION_CURVES(DIMS, ACCURACY, AUC, SERIESNAMES, NUMFOLDS,
%   FILENAME, LABELS) plots the fold means of ACCURACY and AUC
%   (numFolds-by-numel(DIMS)-by-numel(SERIESNAMES)) in two panels, one
%   curve per series, and exports the figure to FILENAME. LABELS has fields
%     methodName    e.g. 'PCA + Bayes' (used in the titles)
%     dimSymbol     e.g. 'd''' or 'p' (used in the titles)
%     dimAxisLabel  x-axis label

fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 420]);
if exist('theme', 'file')
    % Light figure for the report, even when MATLAB uses a dark theme.
    theme(fig, 'light');
end
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
values = {auc, accuracy};
metricNames = {'AUC', 'Accuracy'};
for q = 1:2
    nexttile(layout);
    hold on;
    for t = 1:numel(seriesNames)
        plot(dims, mean(values{q}(:, :, t), 1), 'LineWidth', 1.5, ...
            'DisplayName', seriesNames{t});
    end
    hold off;
    xlabel(labels.dimAxisLabel);
    ylabel(sprintf('%s (mean of %d folds)', metricNames{q}, numFolds));
    title(sprintf('%s: %s vs. %s', labels.methodName, metricNames{q}, ...
        labels.dimSymbol));
    xlim([min(dims), max(dims)]);
    % Zoom in on the range the curves actually cover.
    lowest = min(mean(values{q}, 1), [], 'all');
    ylim([min(0.9, floor(lowest * 20) / 20), 1.005]);
    legend('Location', 'southwest');
    grid on;
end
exportgraphics(fig, fileName, 'Resolution', 150);
close(fig);
end
