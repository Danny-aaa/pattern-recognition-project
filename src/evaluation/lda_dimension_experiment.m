% 【檔案功能】第十一週進度：LDA + Bayes 的五折交叉驗證，並觀察維度對結果的影響。
%   兩類問題的 LDA 維度上限是 c - 1 = 1（S_B 的秩為 1），輸出維度無法再調整，
%   所以這裡調整的是 LDA 之前的 PCA 維度 p（Fisherface 作法）：
%     影像 d 維 --PCA--> p 維 --LDA--> 1 維 --> 貝氏分類器
%   p 的上限是 n - c = n - 2，超過後類內散佈矩陣 S_W 會變成奇異。
%   對每一個 p 記錄準確率、ROC 與 AUC，畫出 AUC（與準確率）對 p 的曲線，
%   並用數值驗證 S_W^-1 S_B 只有一個非零特徵值。
%   Fisher 方向預設用 Ledoit-Wolf 收縮後的 S_W 計算（見 fit_fisher）。
%   貝氏分類器預設用內層交叉驗證得到的折外 LDA 特徵訓練（見
%   lda_out_of_fold），不用被 Fisher 方向看過的訓練投影，避免低估變異數。
%   另外記錄測試折投影與貝氏分類器訓練資料的類內變異數比值（比值遠大於 1
%   表示貝氏分類器低估了變異數，接近 1 表示估得準）。
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
%   (mean and std over folds, and of the pooled out-of-fold scores), the
%   training accuracy, and the variance ratio described below. DETAILS
%   holds the per-fold values, the out-of-fold scores, the rank check of
%   S_B, and the options.
%
%   The Fisher direction over-separates the training samples it was fitted
%   on, so a Bayes classifier fitted on those projections underestimates
%   the class variances. With OPTIONS.innerFolds > 1 the training folds
%   are split again by person and the Bayes classifier is fitted on the
%   out-of-fold LDA features of LDA_OUT_OF_FOLD instead; the test fold is
%   projected with the Fisher direction of all training folds.
%   VarianceRatio is the pooled within-class variance of the LDA feature on
%   the test fold divided by that of the data the Bayes classifier was
%   fitted on. A ratio far above 1 means the variances are underestimated
%   and the decision threshold is too narrow; about 1 means they match.
%   DETAILS.varianceRatioInSample always divides by the in-sample
%   training projections.
%
%   In each fold PCA is fitted once on the training folds; the first p
%   columns of the projection are the p-dimensional PCA features, in which
%   FIT_FISHER gives the Fisher direction.
%
%   OPTIONS fields (all optional):
%     dims             []      PCA dimensions to test; default 1:(n_train - 2)
%     covarianceTypes  {'full', 'shared'}  Bayes covariance cases compared
%                              (see FIT_BAYES)
%     fisherShrinkage  'ledoit-wolf'  S_W estimate of FIT_FISHER ('none' is
%                              the textbook S_W^-1)
%     innerFolds       5       inner person-grouped folds for the
%                              out-of-fold Bayes training features; 0 or 1
%                              fits the Bayes classifier on the in-sample
%                              projections (the old behavior). Without
%                              shrinkage, p must stay below n_inner - 2.
%     regularization   'ledoit-wolf'  Bayes covariance estimate (see
%                              FIT_BAYES; for the 1-D LDA feature both
%                              choices are practically the plain variance)
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
    'fisherShrinkage', 'ledoit-wolf', 'innerFolds', 5, ...
    'regularization', 'ledoit-wolf', 'numFolds', 5, 'seed', 1, ...
    'rocDims', [1, 5, 20, 50, 100, 150], ...
    'plotFile', fullfile(paths.results, 'lda_dimension_auc.png'), ...
    'rocPlotFile', fullfile(paths.results, 'lda_dimension_roc.png'), ...
    'resultFile', fullfile(paths.results, 'lda_dimension_results.mat'), ...
    'verbose', true);
options = merge_options(options, defaults);
options.covarianceTypes = validate_covariance_types(options.covarianceTypes);

if ~isfile(datasetFile)
    error('Training-vector file not found: %s', datasetFile);
end
data = load(datasetFile, 'X', 'y', 'fileNames');
X = data.X;
y = data.y(:);
fileNames = data.fileNames(:);
foldId = make_person_folds(fileNames, y, options.numFolds, options.seed);
clear data
useInnerFolds = options.innerFolds > 1;

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
fisherShrinkage = zeros(numFolds, numDims);
varianceRatio = zeros(numFolds, numDims);
varianceRatioInSample = zeros(numFolds, numDims);
rankSb = zeros(numFolds, numDims);
numDiscriminants = zeros(numFolds, numDims);

for k = 1:numFolds
    isTest = (foldId == k);
    ytr = y(~isTest);
    yte = y(isTest);

    tic;
    [pcaModel, Atr] = fit_pca(X(~isTest, :), max(dims));
    Ate = project_features(pcaModel, X(isTest, :));
    pcaTime = toc;
    clear pcaModel

    % Out-of-fold LDA features of the training folds for the Bayes fit.
    tic;
    if useInnerFolds
        innerFoldId = make_person_folds(fileNames(~isTest), ytr, ...
            options.innerFolds, options.seed);
        zOutOfFold = lda_out_of_fold(Atr, ytr, innerFoldId, dims, ...
            options.fisherShrinkage);
    end
    innerTime = toc;

    for j = 1:numDims
        p = dims(j);
        fisher = fit_fisher(Atr(:, 1:p), ytr, options.fisherShrinkage);
        ztr = Atr(:, 1:p) * fisher.w;
        zte = Ate(:, 1:p) * fisher.w;
        if useInnerFolds
            zBayes = zOutOfFold(:, j);
        else
            zBayes = ztr;
        end
        fisherCriterion(k, j) = fisher.fisherCriterion;
        fisherShrinkage(k, j) = fisher.shrinkage;
        varianceRatio(k, j) = withinClassVariance(zte, yte) ...
            / withinClassVariance(zBayes, ytr);
        varianceRatioInSample(k, j) = withinClassVariance(zte, yte) ...
            / withinClassVariance(ztr, ytr);

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
            classifier = fit_bayes(zBayes, ytr, types{t}, options.regularization);
            [yPred, score] = predict_bayes(classifier, zte);
            scores(isTest, j, t) = score;
            accuracy(k, j, t) = mean(yPred == yte);
            trainAccuracy(k, j, t) = mean(predict_bayes(classifier, ztr) == ytr);
            [~, ~, auc(k, j, t)] = roc_auc(score, yte);
        end
    end
    if options.verbose
        fprintf(['Fold %d/%d done (PCA %.1f s, inner CV %.1f s, ' ...
            '%d dimensions x %d covariance types)\n'], ...
            k, numFolds, pcaTime, innerTime, numDims, numTypes);
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
        mean(varianceRatio, 1).', mean(fisherShrinkage, 1).', ...
        'VariableNames', {'Covariance', 'PcaDim', 'LdaDim', ...
        'AccuracyMean', 'AccuracyStd', 'AUCMean', 'AUCStd', 'AUCPooled', ...
        'TrainAccuracyMean', 'VarianceRatio', 'FisherShrinkage'});
    summary = [summary; part]; %#ok<AGROW>
end

details = struct('options', options, 'dims', dims, ...
    'covarianceTypes', {types}, 'foldId', foldId, 'accuracy', accuracy, ...
    'trainAccuracy', trainAccuracy, 'auc', auc, 'aucPooled', aucPooled, ...
    'scores', scores, 'y', y, 'fisherCriterion', fisherCriterion, ...
    'fisherShrinkage', fisherShrinkage, 'varianceRatio', varianceRatio, ...
    'varianceRatioInSample', varianceRatioInSample, ...
    'rankSb', rankSb, 'numDiscriminants', numDiscriminants, ...
    'ldaDimensionLimit', numClasses - 1, 'maxPcaDim', maxDim);

if options.verbose
    fprintf(['LDA dimension upper bound c - 1 = %d; rank(S_B) over all ' ...
        'folds and p: min %d, max %d\n'], numClasses - 1, ...
        min(rankSb, [], 'all'), max(rankSb, [], 'all'));
    fprintf('Nonzero eigenvalues of S_W^-1 S_B: min %d, max %d\n', ...
        min(numDiscriminants, [], 'all'), max(numDiscriminants, [], 'all'));
    for t = 1:numTypes
        meanAuc = mean(auc(:, :, t), 1);
        meanAcc = mean(accuracy(:, :, t), 1);
        [bestAuc, iAuc] = max(meanAuc);
        [bestAcc, iAcc] = max(meanAcc);
        [worstAcc, iWorst] = min(meanAcc);
        fprintf(['%-6s covariance: best mean AUC = %.4f at p = %d, ' ...
            'best mean accuracy = %.3f at p = %d, worst = %.3f at p = %d\n'], ...
            types{t}, bestAuc, dims(iAuc), bestAcc, dims(iAcc), ...
            worstAcc, dims(iWorst));
    end
end

if ~isempty(options.resultFile)
    save(options.resultFile, 'summary', 'details', '-v7');
end
seriesNames = cellfun(@(t) covariance_label(t, 1), types, ...
    'UniformOutput', false);
labels = struct('methodName', 'LDA + Bayes', 'dimSymbol', 'p', ...
    'dimAxisLabel', 'PCA dimension p before LDA (LDA output: 1 dimension)');
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

function v = withinClassVariance(z, y)
% Pooled within-class variance of a 1-D feature: sum_i sum (z - c_i)^2 / n.
v = 0;
for label = [0, 1]
    zi = z(y == label);
    v = v + sum((zi - mean(zi)) .^ 2);
end
v = v / numel(z);
end
