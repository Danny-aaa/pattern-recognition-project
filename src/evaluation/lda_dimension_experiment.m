% 【檔案功能】第十一週進度：LDA + Bayes 的五折交叉驗證，並觀察維度對結果的影響。
%   兩類問題的 LDA 維度上限是 c - 1 = 1（S_B 的秩為 1），輸出維度無法再調整，
%   所以這裡調整的是 LDA 之前的 PCA 維度 p（Fisherface 作法）：
%     影像 d 維 --PCA--> p 維 --LDA--> 1 維 --> 貝氏分類器
%   p 的上限是 n - c = n - 2，超過後類內散佈矩陣 S_W 會變成奇異。
%   對每一個 p 記錄準確率、ROC 與 AUC，畫出 AUC（與準確率）對 p 的曲線，
%   並用數值驗證 S_W^-1 S_B 只有一個非零特徵值。
%   輸出 results/lda_dimension_results.mat、lda_dimension_auc.png、
%   lda_dimension_roc.png。
%   參考文獻：P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman,
%   "Eigenfaces vs. Fisherfaces: Recognition using class specific linear
%   projection," IEEE TPAMI, vol. 19, no. 7, pp. 711-720, 1997.

function [summary, details] = lda_dimension_experiment(datasetFile, options)
%LDA_DIMENSION_EXPERIMENT Effect of the dimension on LDA + Bayes.
%   [SUMMARY, DETAILS] = LDA_DIMENSION_EXPERIMENT(DATASETFILE, OPTIONS) runs
%   person-grouped K-fold cross-validation of LDA + Bayes. The LDA output
%   has c - 1 = 1 dimension for two classes, so the dimension that is
%   varied is the PCA dimension p used before LDA (OPTIONS.dims). SUMMARY
%   is a table with one row per p: accuracy (mean and std over folds), AUC
%   (mean and std over folds, and of the pooled out-of-fold scores), and
%   the training accuracy. DETAILS holds the per-fold values, the
%   out-of-fold scores, the rank check of S_B, and the options.
%
%   In each fold PCA is fitted once on the training folds; the first p
%   columns of the projection are the p-dimensional PCA features, in which
%   FIT_FISHER gives the Fisher direction.
%
%   OPTIONS fields (all optional):
%     dims             []      PCA dimensions to test; default 1:(n_train - 2)
%     covarianceTypes  {'full', 'shared'}  Bayes covariance cases compared
%                              (see FIT_BAYES)
%     regularization   1e-6    Bayes covariance regularization
%     numFolds         5
%     seed             1       fold assignment seed
%     rocDims          [1 5 20 50 100 150]  dimensions shown in the ROC figure
%     plotFile         'results/lda_dimension_auc.png' ('' to skip)
%     rocPlotFile      'results/lda_dimension_roc.png' ('' to skip)
%     resultFile       'results/lda_dimension_results.mat' ('' to skip)
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
    'rocDims', [1, 5, 20, 50, 100, 150], ...
    'plotFile', fullfile(paths.results, 'lda_dimension_auc.png'), ...
    'rocPlotFile', fullfile(paths.results, 'lda_dimension_roc.png'), ...
    'resultFile', fullfile(paths.results, 'lda_dimension_results.mat'), ...
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
numClasses = 2;
% S_W has rank at most n_train - c, so p cannot exceed n_train - 2.
maxDim = inf;
for k = 1:numFolds
    maxDim = min(maxDim, nnz(foldId ~= k) - numClasses);
end
if isempty(options.dims)
    options.dims = 1:maxDim;
end
dims = options.dims(:).';
if any(dims < 1) || any(dims > maxDim) || any(dims ~= round(dims))
    error('dims must be integers between 1 and %d (n_train - 2).', maxDim);
end

types = options.covarianceTypes;
numDims = numel(dims);
numTypes = numel(types);
accuracy = zeros(numFolds, numDims, numTypes);
trainAccuracy = zeros(numFolds, numDims, numTypes);
auc = zeros(numFolds, numDims, numTypes);
scores = zeros(numel(y), numDims, numTypes);
fisherCriterion = zeros(numFolds, numDims);
rankSb = zeros(numFolds, numDims);
numDiscriminants = zeros(numFolds, numDims);

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

    for j = 1:numDims
        p = dims(j);
        fisher = fit_fisher(Atr(:, 1:p), ytr);
        ztr = Atr(:, 1:p) * fisher.w;
        zte = Ate(:, 1:p) * fisher.w;
        fisherCriterion(k, j) = fisher.fisherCriterion;

        % LDA dimension limit: S_W^-1 S_B has rank(S_B) <= c - 1 nonzero
        % eigenvalues, so only one discriminant direction exists.
        rankSb(k, j) = rank(fisher.Sb);
        if fisher.rcondSw > 1e-12
            lambda = sort(abs(eig(fisher.Sb, fisher.Sw)), 'descend');
            numDiscriminants(k, j) = nnz(lambda > 1e-8 * lambda(1));
        else
            numDiscriminants(k, j) = NaN;
        end

        for t = 1:numTypes
            classifier = fit_bayes(ztr, ytr, types{t}, options.regularization);
            [yPred, score] = predict_bayes(classifier, zte);
            scores(isTest, j, t) = score;
            accuracy(k, j, t) = mean(yPred == yte);
            trainAccuracy(k, j, t) = mean(predict_bayes(classifier, ztr) == ytr);
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
        ones(numDims, 1), ...
        mean(accuracy(:, :, t), 1).', std(accuracy(:, :, t), 0, 1).', ...
        mean(auc(:, :, t), 1).', std(auc(:, :, t), 0, 1).', ...
        aucPooled(:, t), mean(trainAccuracy(:, :, t), 1).', ...
        'VariableNames', {'Covariance', 'PcaDim', 'LdaDim', ...
        'AccuracyMean', 'AccuracyStd', 'AUCMean', 'AUCStd', 'AUCPooled', ...
        'TrainAccuracyMean'});
    summary = [summary; part]; %#ok<AGROW>
end

details = struct('options', options, 'dims', dims, ...
    'covarianceTypes', {types}, 'foldId', foldId, 'accuracy', accuracy, ...
    'trainAccuracy', trainAccuracy, 'auc', auc, 'aucPooled', aucPooled, ...
    'scores', scores, 'y', y, 'fisherCriterion', fisherCriterion, ...
    'rankSb', rankSb, 'numDiscriminants', numDiscriminants, ...
    'ldaDimensionLimit', numClasses - 1, 'maxPcaDim', maxDim);

if options.verbose
    fprintf('LDA dimension upper bound c - 1 = %d; rank(S_B) = %d in every fold and p: %d\n', ...
        numClasses - 1, numClasses - 1, all(rankSb == numClasses - 1, 'all'));
    fprintf('Nonzero eigenvalues of S_W^-1 S_B: min %d, max %d\n', ...
        min(numDiscriminants, [], 'all'), max(numDiscriminants, [], 'all'));
    for t = 1:numTypes
        meanAuc = mean(auc(:, :, t), 1);
        meanAcc = mean(accuracy(:, :, t), 1);
        [bestAuc, iAuc] = max(meanAuc);
        [bestAcc, iAcc] = max(meanAcc);
        fprintf(['%-6s covariance: best mean AUC = %.4f at p = %d, ' ...
            'best mean accuracy = %.3f at p = %d\n'], types{t}, bestAuc, ...
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
% AUC and accuracy against the PCA dimension p used before LDA.
fig = figure('Visible', 'off', 'Position', [0, 0, 1100, 420]);
if exist('theme', 'file')
    theme(fig, 'light');
end
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
values = {auc, accuracy};
labels = {'AUC', 'Accuracy'};
for q = 1:2
    nexttile(layout);
    hold on;
    for t = 1:numel(types)
        plot(dims, mean(values{q}(:, :, t), 1), 'LineWidth', 1.5, ...
            'DisplayName', covarianceLabel(types{t}));
    end
    hold off;
    xlabel('PCA dimension p before LDA (LDA output: 1 dimension)');
    ylabel(sprintf('%s (mean of %d folds)', labels{q}, numFolds));
    title(sprintf('LDA + Bayes: %s vs. dimension', labels{q}));
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
    for p = rocDims
        [falseAlarm, hit, areaUnder] = roc_auc(scores(:, dims == p, t), y);
        plot(falseAlarm, hit, 'LineWidth', 1.3, 'DisplayName', ...
            sprintf('p = %d (AUC = %.4f)', p, areaUnder));
    end
    plot([0, 1], [0, 1], 'k:', 'HandleVisibility', 'off');
    hold off;
    axis square;
    xlabel('False alarm rate P(score > x^* | man)');
    ylabel('Hit rate P(score > x^* | woman)');
    title(sprintf('%d-fold ROC, LDA + %s', numFolds, covarianceLabel(types{t})));
    legend('Location', 'southeast');
    grid on;
end
exportgraphics(fig, fileName, 'Resolution', 150);
close(fig);
end

function label = covarianceLabel(covarianceType)
if strcmp(covarianceType, 'full')
    label = 'Bayes, \sigma_i^2 separate';
else
    label = 'Bayes, \sigma^2 shared';
end
end
