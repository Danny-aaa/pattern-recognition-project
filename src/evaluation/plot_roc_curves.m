% 【檔案功能】評估：畫出幾個維度下合併各折分數的 ROC 曲線，存成圖檔。
%   PCA 維度實驗與 LDA 維度實驗共用這個函式。

function plot_roc_curves(dims, scores, y, seriesNames, rocDims, numFolds, ...
    fileName, labels)
%PLOT_ROC_CURVES Pooled out-of-fold ROC curves for a few dimensions.
%   PLOT_ROC_CURVES(DIMS, SCORES, Y, SERIESNAMES, ROCDIMS, NUMFOLDS,
%   FILENAME, LABELS) draws one panel per series. Each panel shows the ROC
%   curve of SCORES(:, j, t) (out-of-fold scores, one column per entry of
%   DIMS) for every dimension in ROCDIMS that is also in DIMS. ROCDIMS may
%   be a row or a column vector. LABELS has the fields methodName and
%   dimSymbol (see PLOT_DIMENSION_CURVES). Nothing is drawn when no
%   dimension of ROCDIMS was tested.

% intersect returns a row only when both inputs are rows; a column here
% would make the for loop below run once with the whole vector.
rocDims = intersect(rocDims(:).', dims(:).');
if isempty(rocDims)
    return;
end
fig = figure('Visible', 'off', ...
    'Position', [0, 0, 520 * numel(seriesNames), 480]);
if exist('theme', 'file')
    theme(fig, 'light');
end
layout = tiledlayout(fig, 1, numel(seriesNames), 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for t = 1:numel(seriesNames)
    nexttile(layout);
    hold on;
    for d = rocDims
        [falseAlarm, hit, areaUnder] = roc_auc(scores(:, dims == d, t), y);
        plot(falseAlarm, hit, 'LineWidth', 1.3, 'DisplayName', ...
            sprintf('%s = %d (AUC = %.4f)', labels.dimSymbol, d, areaUnder));
    end
    plot([0, 1], [0, 1], 'k:', 'HandleVisibility', 'off');
    hold off;
    axis square;
    xlabel('False alarm rate P(score > x^* | man)');
    ylabel('Hit rate P(score > x^* | woman)');
    title(sprintf('%d-fold ROC, %s, %s', numFolds, labels.methodName, ...
        seriesNames{t}));
    legend('Location', 'southeast');
    grid on;
end
exportgraphics(fig, fileName, 'Resolution', 150);
close(fig);
end
