% 【檔案功能】第十週進度：觀察 PCA 維度 d' 對 PCA + Bayes 實驗結果的影響。
%   用依人分組的五折交叉驗證，對每一個 PCA 維度 d' 訓練貝氏分類器，
%   記錄準確率、ROC 與 AUC，畫出 AUC（與準確率）對 d' 的曲線。
%   每一折只做一次 PCA（取到上限 n - 1 維），前 d' 個主成分就是 d' 維的
%   PCA 特徵，所以不必對每個 d' 重新計算特徵向量。
%   貝氏分類器的共變異數預設用 Ledoit-Wolf 收縮估計（見 fit_bayes），
%   d' 接近或超過每類訓練樣本數（約 80）時才不會因矩陣奇異而失準。
%   輸出 results/pca_dimension_results.mat、pca_dimension_auc.png、
%   pca_dimension_roc.png。

function [summary, details] = pca_dimension_experiment(datasetFile, options)
%PCA_DIMENSION_EXPERIMENT Effect of the PCA dimension d' on PCA + Bayes.
%   [SUMMARY, DETAILS] = PCA_DIMENSION_EXPERIMENT(DATASETFILE, OPTIONS) runs
%   person-grouped K-fold cross-validation of PCA + Bayes for every PCA
%   dimension in OPTIONS.dims. SUMMARY is a table with one row per d':
%   accuracy (mean and std over folds), AUC (mean and std over folds, and
%   of the pooled out-of-fold scores), and the mean Ledoit-Wolf shrinkage
%   of the Bayes covariances. DETAILS holds the per-fold values, the
%   out-of-fold scores, and the options.
%
%   In each fold PCA is fitted once on the training folds with all n - 1
%   components. The eigenvectors are sorted by eigenvalue, so the first d'
%   columns of the projection are exactly the d'-dimensional PCA features.
%
%   OPTIONS fields (all optional):
%     dims             []      PCA dimensions to test; default 1:(n_train - 1)
%     covarianceTypes  {'full', 'shared'}  Bayes covariance cases compared
%                              (see FIT_BAYES)
%     regularization   'ledoit-wolf'  Bayes covariance estimate (see
%                              FIT_BAYES); a scalar such as 1e-6 gives the
%                              old fixed ridge
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
    'regularization', 'ledoit-wolf', 'numFolds', 5, 'seed', 1, ...
    'rocDims', [1, 5, 10, 20, 50, 100], ...
    'plotFile', fullfile(paths.results, 'pca_dimension_auc.png'), ...
    'rocPlotFile', fullfile(paths.results, 'pca_dimension_roc.png'), ...
    'resultFile', fullfile(paths.results, 'pca_dimension_results.mat'), ...
    'verbose', true);
options = merge_options(options, defaults);
options.covarianceTypes = validate_covariance_types(options.covarianceTypes);

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
shrinkage = nan(numFolds, numDims, numTypes);
scores = zeros(numel(y), numDims, numTypes);

for k = 1:numFolds
    isTest = (foldId == k);
    ytr = y(~isTest);
    yte = y(isTest);

    tic;
    [pcaModel, Atr] = fit_pca(X(~isTest, :), max(dims));
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
            shrinkage(k, j, t) = mean(classifier.shrinkage);
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
        aucPooled(:, t), mean(shrinkage(:, :, t), 1).', ...
        'VariableNames', {'Covariance', 'PcaDim', 'AccuracyMean', ...
        'AccuracyStd', 'AUCMean', 'AUCStd', 'AUCPooled', 'ShrinkageMean'});
    summary = [summary; part]; %#ok<AGROW>
end

details = struct('options', options, 'dims', dims, ...
    'covarianceTypes', {types}, 'foldId', foldId, 'accuracy', accuracy, ...
    'auc', auc, 'aucPooled', aucPooled, 'shrinkage', shrinkage, ...
    'scores', scores, 'y', y, 'maxDim', maxDim);

if options.verbose
    for t = 1:numTypes
        meanAuc = mean(auc(:, :, t), 1);
        meanAcc = mean(accuracy(:, :, t), 1);
        [bestAuc, iAuc] = max(meanAuc);
        [bestAcc, iAcc] = max(meanAcc);
        [worstAcc, iWorst] = min(meanAcc);
        fprintf(['%-6s covariance: best mean AUC = %.4f at d'' = %d, ' ...
            'best mean accuracy = %.3f at d'' = %d, worst = %.3f at d'' = %d\n'], ...
            types{t}, bestAuc, dims(iAuc), bestAcc, dims(iAcc), ...
            worstAcc, dims(iWorst));
    end
end

if ~isempty(options.resultFile)
    save(options.resultFile, 'summary', 'details', '-v7');
end
seriesNames = cellfun(@(t) covariance_label(t, 2), types, ...
    'UniformOutput', false);
labels = struct('methodName', 'PCA + Bayes', 'dimSymbol', 'd''', ...
    'dimAxisLabel', 'PCA dimension d''');
if ~isempty(options.plotFile)
    plot_dimension_curves(dims, accuracy, auc, seriesNames, numFolds, ...
        options.plotFile, labels);
end
if ~isempty(options.rocPlotFile)
    plot_roc_curves(dims, scores, y, seriesNames, options.rocDims, ...
        numFolds, options.rocPlotFile, labels);
end
if options.verbose
    saved = {options.resultFile, options.plotFile, options.rocPlotFile};
    saved = saved(~cellfun(@isempty, saved));
    if ~isempty(saved)
        fprintf('Saved %s\n', strjoin(saved, ', '));
    end
end
end
