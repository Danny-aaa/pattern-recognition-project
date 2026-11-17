% 【檔案功能】第十週進度：觀察 PCA 維度 d' 對 PCA + Bayes 實驗結果的影響。
%   用依人分組的五折交叉驗證，對每一個 PCA 維度 d' 訓練貝氏分類器，
%   記錄準確率、ROC 與 AUC，畫出 AUC（與準確率）對 d' 的曲線。
%   每一折只做一次 PCA（取到上限 n - 1 維），前 d' 個主成分就是 d' 維的
%   PCA 特徵，所以不必對每個 d' 重新計算特徵向量。
%   輸出 results/pca_dimension_results.mat、pca_dimension_auc.png、
%   pca_dimension_roc.png。

function [summary, details] = pca_dimension_experiment(datasetFile, options)
%PCA_DIMENSION_EXPERIMENT Effect of the PCA dimension d' on PCA + Bayes.
%   [SUMMARY, DETAILS] = PCA_DIMENSION_EXPERIMENT(DATASETFILE, OPTIONS) runs
%   person-grouped K-fold cross-validation of PCA + Bayes for every PCA
%   dimension in OPTIONS.dims. SUMMARY is a table with one row per d':
%   accuracy (mean and std over folds), AUC (mean and std over folds, and
%   of the pooled out-of-fold scores). DETAILS holds the per-fold values,
%   the out-of-fold scores, and the options.
%
%   In each fold PCA is fitted once on the training folds with all n - 1
%   components. The eigenvectors are sorted by eigenvalue, so the first d'
%   columns of the projection are exactly the d'-dimensional PCA features.
%
%   OPTIONS fields (all optional):
%     dims             []      PCA dimensions to test; default 1:(n_train - 1)
%     covarianceTypes  {'full', 'shared'}  Bayes covariance cases compared
%                              (see FIT_BAYES)
%     regularization   1e-6    Bayes covariance regularization
%     numFolds         5
%     seed             1       fold assignment seed
%     rocDims          [1 5 10 20 50 100]  dimensions shown in the ROC figure
%     plotFile         'results/pca_dimension_auc.png' ('' to skip)
%     rocPlotFile      'results/pca_dimension_roc.png' ('' to skip)
%     resultFile       'results/pca_dimension_results.mat' ('' to skip)
%     verbose          true

paths = project_paths();
if nargin < 1 || isempty(datasetFile)
    datasetFile = fullfile(paths.data, 'training_vectors_640x640x3_RGB.mat');
end
if nargin < 2
    options = struct();
end
defaults = struct('dims', [], 'covarianceTypes', {{'full', 'shared'}}, ...
    'regularization', 1e-6, 'numFolds', 5, 'seed', 1, ...
    'rocDims', [1, 5, 10, 20, 50, 100], ...
    'plotFile', fullfile(paths.results, 'pca_dimension_auc.png'), ...
    'rocPlotFile', fullfile(paths.results, 'pca_dimension_roc.png'), ...
    'resultFile', fullfile(paths.results, 'pca_dimension_results.mat'), ...
    'verbose', true);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(options, names{i})
        options.(names{i}) = defaults.(names{i});
    end
end
if ischar(options.covarianceTypes) || isstring(options.covarianceTypes)
    options.covarianceTypes = cellstr(options.covarianceTypes);
end

if ~isfile(datasetFile)
    error('Training-vector file not found: %s', datasetFile);
end
data = load(datasetFile, 'X', 'y', 'fileNames');
X = data.X;
y = data.y(:);
foldId = make_person_folds(data.fileNames, y, options.numFolds, options.seed);
clear data

numFolds = options.numFolds;
% PCA dimension limit of the smallest training fold (n_train - 1).
maxDim = inf;
for k = 1:numFolds
    maxDim = min(maxDim, nnz(foldId ~= k) - 1);
end
if isempty(options.dims)
    options.dims = 1:maxDim;
end
dims = options.dims(:).';
if any(dims < 1) || any(dims > maxDim) || any(dims ~= round(dims))
    error('dims must be integers between 1 and %d (n_train - 1).', maxDim);
end

types = options.covarianceTypes;
numDims = numel(dims);
numTypes = numel(types);
accuracy = zeros(numFolds, numDims, numTypes);
auc = zeros(numFolds, numDims, numTypes);
scores = zeros(numel(y), numDims, numTypes);

for k = 1:numFolds
    isTest = (foldId == k);
    ytr = y(~isTest);
    yte = y(isTest);

    tic;
    pcaModel = fit_pca(X(~isTest, :), max(dims));
    Atr = project_features(pcaModel, X(~isTest, :));
    Ate = project_features(pcaModel, X(isTest, :));
    pcaTime = toc;
    clear pcaModel

    for t = 1:numTypes
        for j = 1:numDims
            d = dims(j);
            classifier = fit_bayes(Atr(:, 1:d), ytr, types{t}, ...
                options.regularization);
            [yPred, score] = predict_bayes(classifier, Ate(:, 1:d));
            scores(isTest, j, t) = score;
            accuracy(k, j, t) = mean(yPred == yte);
            [~, ~, auc(k, j, t)] = roc_auc(score, yte);
        end
    end
    if options.verbose
        fprintf('Fold %d/%d done (PCA %.1f s, %d dimensions x %d covariance types)\n', ...
            k, numFolds, pcaTime, numDims, numTypes);
    end
end

% Pooled out-of-fold AUC for every dimension.
aucPooled = zeros(numDims, numTypes);
for t = 1:numTypes
    for j = 1:numDims
        [~, ~, aucPooled(j, t)] = roc_auc(scores(:, j, t), y);
    end
end

summary = table();
for t = 1:numTypes
    part = table( ...
        repmat(string(types{t}), numDims, 1), dims.', ...
        mean(accuracy(:, :, t), 1).', std(accuracy(:, :, t), 0, 1).', ...
        mean(auc(:, :, t), 1).', std(auc(:, :, t), 0, 1).', ...
        aucPooled(:, t), ...
        'VariableNames', {'Covariance', 'PcaDim', 'AccuracyMean', ...
        'AccuracyStd', 'AUCMean', 'AUCStd', 'AUCPooled'});
    summary = [summary; part]; %#ok<AGROW>
end

details = struct('options', options, 'dims', dims, ...
    'covarianceTypes', {types}, 'foldId', foldId, 'accuracy', accuracy, ...
    'auc', auc, 'aucPooled', aucPooled, 'scores', scores, 'y', y, ...
    'maxDim', maxDim);

if options.verbose
    for t = 1:numTypes
        meanAuc = mean(auc(:, :, t), 1);
        meanAcc = mean(accuracy(:, :, t), 1);
        [bestAuc, iAuc] = max(meanAuc);
        [bestAcc, iAcc] = max(meanAcc);
        fprintf(['%-6s covariance: best mean AUC = %.4f at d'' = %d, ' ...
            'best mean accuracy = %.3f at d'' = %d\n'], types{t}, bestAuc, ...
            dims(iAuc), bestAcc, dims(iAcc));
    end
end

if ~isempty(options.resultFile)
    save(options.resultFile, 'summary', 'details', '-v7');
end
if ~isempty(options.plotFile)
    plotDimensionCurves(dims, accuracy, auc, types, numFolds, options.plotFile);
end
if ~isempty(options.rocPlotFile)
    plotRocCurves(dims, scores, y, types, options.rocDims, numFolds, ...
        options.rocPlotFile);
end
if options.verbose
    fprintf('Saved results to %s\n', paths.results);
end
end

function plotDimensionCurves(dims, accuracy, auc, types, numFolds, fileName)
% AUC and accuracy against the PCA dimension d' (mean over folds).
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 420]);
if exist('theme', 'file')
    theme(fig, 'light');
end
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
values = {auc, accuracy};
labels = {'AUC', 'Accuracy'};
for p = 1:2
    nexttile(layout);
    hold on;
    for t = 1:numel(types)
        plot(dims, mean(values{p}(:, :, t), 1), 'LineWidth', 1.5, ...
            'DisplayName', covarianceLabel(types{t}));
    end
    hold off;
    xlabel('PCA dimension d''');
    ylabel(sprintf('%s (mean of %d folds)', labels{p}, numFolds));
    title(sprintf('PCA + Bayes: %s vs. d''', labels{p}));
    xlim([min(dims), max(dims)]);
    % Zoom in on the range the curves actually cover.
    lowest = min(mean(values{p}, 1), [], 'all');
    ylim([min(0.9, floor(lowest * 20) / 20), 1.005]);
    legend('Location', 'southwest');
    grid on;
end
exportgraphics(fig, fileName, 'Resolution', 150);
close(fig);
end

function plotRocCurves(dims, scores, y, types, rocDims, numFolds, fileName)
% Pooled out-of-fold ROC curves for a few PCA dimensions.
rocDims = intersect(rocDims, dims);
if isempty(rocDims)
    return;
end
fig = figure('Visible', 'off', 'Position', [0, 0, 520 * numel(types), 480]);
if exist('theme', 'file')
    theme(fig, 'light');
end
layout = tiledlayout(fig, 1, numel(types), 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for t = 1:numel(types)
    nexttile(layout);
    hold on;
    for d = rocDims
        [falseAlarm, hit, areaUnder] = roc_auc(scores(:, dims == d, t), y);
        plot(falseAlarm, hit, 'LineWidth', 1.3, 'DisplayName', ...
            sprintf('d'' = %d (AUC = %.4f)', d, areaUnder));
    end
    plot([0, 1], [0, 1], 'k:', 'HandleVisibility', 'off');
    hold off;
    axis square;
    xlabel('False alarm rate P(score > x^* | man)');
    ylabel('Hit rate P(score > x^* | woman)');
    title(sprintf('%d-fold ROC, %s', numFolds, covarianceLabel(types{t})));
    legend('Location', 'southeast');
    grid on;
end
exportgraphics(fig, fileName, 'Resolution', 150);
close(fig);
end

function label = covarianceLabel(covarianceType)
if strcmp(covarianceType, 'full')
    label = 'Bayes, \Sigma_i arbitrary (quadratic)';
else
    label = 'Bayes, \Sigma_i = \Sigma (linear)';
end
end
