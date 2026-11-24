% 【檔案功能】評估：對 PCA + Bayes 與 LDA + Bayes 進行依人分組的五折交叉驗證。
%   每一折都只用訓練折來訓練降維與分類器，再套用到測試折。輸出各方法的
%   準確率、AUC、特徵維度、參數量、訓練時間與推論時間，並把合併後的
%   ROC 曲線存成 results/cv_roc_curves.png、數值存成 results/cv_results.mat。
%   每一折只做一次 PCA，PCA + Bayes 取前 d' 個主成分，LDA + Bayes 取前 p 個
%   主成分再求 Fisher 方向，避免重複計算 160 x 1,228,800 的 Gram 矩陣。

function [results, details] = run_cross_validation(datasetFile, options)
%RUN_CROSS_VALIDATION Person-grouped 5-fold cross-validation of the classifiers.
%   [RESULTS, DETAILS] = RUN_CROSS_VALIDATION(DATASETFILE, OPTIONS) loads
%   the training vectors created by BUILD_TRAINING_VECTORS and evaluates
%     PCA + Bayes
%     LDA + Bayes
%   with K-fold cross-validation. In every fold, PCA/LDA and the
%   classifier are fitted on the training folds only and applied to the
%   held-out fold. RESULTS is a table with one row per method:
%     accuracy (mean and std over folds), AUC (mean over folds and of the
%     pooled out-of-fold scores), feature dimension, parameter counts,
%     training time per fold, and inference time per image.
%   DETAILS holds the per-fold values and the pooled ROC curves.
%
%   PCA is fitted once per fold (all n_train - 1 components) and both
%   methods use its leading components: the first d' for PCA + Bayes, the
%   first p for the Fisher direction of LDA + Bayes. TrainTime_s of each
%   method therefore includes the shared PCA time; inference projects the
%   test images with each method's own projection matrix.
%
%   The Bayes classifier of LDA + Bayes is fitted on out-of-fold LDA
%   features from an inner person-grouped cross-validation of the training
%   folds (LDA_OUT_OF_FOLD): the in-sample projections are over-separated
%   by the Fisher direction and would make the variances too small.
%
%   OPTIONS fields (all optional):
%     numFolds          5
%     seed              1      fold assignment seed
%     pcaComponents     20     d' for PCA + Bayes (see FIT_PCA)
%     ldaPcaComponents  []     PCA dimension p inside LDA; [] uses the
%                              Fisherface maximum n_train - 2. With the
%                              Ledoit-Wolf S_W of FIT_FISHER the maximum no
%                              longer over-separates the training folds.
%     fisherShrinkage   'ledoit-wolf'  S_W estimate of FIT_FISHER ('none'
%                              is the textbook S_W^-1)
%     innerFolds        5      inner folds for the LDA + Bayes training
%                              features; 0 or 1 uses the in-sample
%                              projections (the old behavior)
%     covarianceType    'full' Bayes covariance (see FIT_BAYES)
%     regularization    'ledoit-wolf'  Bayes covariance estimate (see
%                              FIT_BAYES; a scalar gives the old ridge)
%     plotFile          'results/cv_roc_curves.png' ('' to skip the figure)
%     resultFile        'results/cv_results.mat' ('' to skip saving)
%     verbose           true

paths = project_paths();
if nargin < 1 || isempty(datasetFile)
    datasetFile = fullfile(paths.data, 'training_vectors_640x640x3_RGB.mat');
end
if nargin < 2
    options = struct();
end
defaults = struct('numFolds', 5, 'seed', 1, 'pcaComponents', 20, ...
    'ldaPcaComponents', [], 'fisherShrinkage', 'ledoit-wolf', ...
    'innerFolds', 5, 'covarianceType', 'full', ...
    'regularization', 'ledoit-wolf', ...
    'plotFile', fullfile(paths.results, 'cv_roc_curves.png'), ...
    'resultFile', fullfile(paths.results, 'cv_results.mat'), 'verbose', true);
options = merge_options(options, defaults);
options.covarianceType = validatestring(options.covarianceType, ...
    {'full', 'shared'});

if ~isfile(datasetFile)
    error('Training-vector file not found: %s', datasetFile);
end
data = load(datasetFile, 'X', 'y', 'fileNames');
X = data.X;
y = data.y(:);
fileNames = data.fileNames(:);
foldId = make_person_folds(fileNames, y, options.numFolds, options.seed);
clear data

methodNames = {'PCA + Bayes', 'LDA + Bayes'};
numMethods = numel(methodNames);
numFolds = options.numFolds;
accuracy = zeros(numFolds, numMethods);
auc = zeros(numFolds, numMethods);
trainTime = zeros(numFolds, numMethods);
inferenceTime = zeros(numFolds, numMethods);
featureDim = zeros(numFolds, numMethods);
pcaDim = zeros(numFolds, numMethods);
projectionParams = zeros(numFolds, numMethods);
classifierParams = zeros(numFolds, numMethods);
scores = zeros(numel(y), numMethods);

for k = 1:numFolds
    isTest = (foldId == k);
    Xtr = X(~isTest, :);
    ytr = y(~isTest);
    Xte = X(isTest, :);
    yte = y(isTest);

    tic;
    [pcaModel, Atr] = fit_pca(Xtr, []);
    pcaTime = toc;
    clear Xtr

    for m = 1:numMethods
        tic;
        switch m
            case 1
                d = numPcaComponents(pcaModel, options.pcaComponents, ...
                    pcaModel.numComponents);
                reducer = struct('mean', pcaModel.mean, ...
                    'W', pcaModel.W(:, 1:d));
                Ztr = Atr(:, 1:d);
            case 2
                d = numPcaComponents(pcaModel, options.ldaPcaComponents, ...
                    numel(ytr) - 2);
                [reducer, Ztr] = lda_from_pca(pcaModel, Atr, ytr, d, ...
                    options.fisherShrinkage);
                if options.innerFolds > 1
                    innerFoldId = make_person_folds(fileNames(~isTest), ...
                        ytr, options.innerFolds, options.seed);
                    Ztr = lda_out_of_fold(Atr, ytr, innerFoldId, d, ...
                        options.fisherShrinkage);
                end
        end
        classifier = fit_bayes(Ztr, ytr, options.covarianceType, ...
            options.regularization);
        trainTime(k, m) = pcaTime + toc;

        tic;
        [yPred, score] = predict_bayes(classifier, ...
            project_features(reducer, Xte));
        inferenceTime(k, m) = toc / numel(yte);

        scores(isTest, m) = score;
        accuracy(k, m) = mean(yPred == yte);
        [~, ~, auc(k, m)] = roc_auc(score, yte);
        featureDim(k, m) = size(reducer.W, 2);
        pcaDim(k, m) = d;
        projectionParams(k, m) = numel(reducer.W) + numel(reducer.mean);
        classifierParams(k, m) = classifier.numParameters;

        if options.verbose
            fprintf('Fold %d/%d  %-12s acc = %.3f  AUC = %.4f  PCA dim = %d  feature dim = %d\n', ...
                k, numFolds, methodNames{m}, accuracy(k, m), auc(k, m), ...
                d, featureDim(k, m));
        end
    end
    clear Xte pcaModel Atr
end

roc = struct('falseAlarm', cell(1, numMethods), 'hit', [], 'auc', []);
for m = 1:numMethods
    [roc(m).falseAlarm, roc(m).hit, roc(m).auc] = roc_auc(scores(:, m), y);
end

results = table( ...
    string(methodNames).', ...
    mean(pcaDim, 1).', mean(featureDim, 1).', ...
    mean(accuracy, 1).', std(accuracy, 0, 1).', ...
    mean(auc, 1).', [roc.auc].', ...
    mean(projectionParams, 1).', mean(classifierParams, 1).', ...
    mean(trainTime, 1).', 1000 * mean(inferenceTime, 1).', ...
    'VariableNames', {'Method', 'PcaDim', 'FeatureDim', 'AccuracyMean', ...
    'AccuracyStd', 'AUCMean', 'AUCPooled', 'ProjectionParams', ...
    'ClassifierParams', 'TrainTime_s', 'InferenceTime_ms'});

details = struct('options', options, 'foldId', foldId, ...
    'methodNames', {methodNames}, 'accuracy', accuracy, 'auc', auc, ...
    'trainTime', trainTime, 'inferenceTime', inferenceTime, ...
    'featureDim', featureDim, 'pcaDim', pcaDim, ...
    'projectionParams', projectionParams, ...
    'classifierParams', classifierParams, 'scores', scores, 'roc', roc);

if options.verbose
    disp(results);
end

if ~isempty(options.resultFile)
    save(options.resultFile, 'results', 'details', '-v7');
end

if ~isempty(options.plotFile)
    fig = figure('Visible', 'off');
    if exist('theme', 'file')
        % Light figure for the report, even when MATLAB uses a dark theme.
        theme(fig, 'light');
    end
    hold on;
    for m = 1:numMethods
        plot(roc(m).falseAlarm, roc(m).hit, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%s (AUC = %.4f)', methodNames{m}, ...
            roc(m).auc));
    end
    plot([0, 1], [0, 1], 'k:', 'HandleVisibility', 'off');
    hold off;
    axis square;
    xlabel('False alarm rate P(score > x^* | man)');
    ylabel('Hit rate P(score > x^* | woman)');
    title(sprintf('%d-fold cross-validation ROC (pooled out-of-fold scores)', ...
        numFolds));
    legend('Location', 'southeast');
    grid on;
    exportgraphics(fig, options.plotFile, 'Resolution', 150);
    close(fig);
    if options.verbose
        fprintf('Saved ROC figure to %s\n', options.plotFile);
    end
end
end

function k = numPcaComponents(pcaModel, requested, maximum)
% PCA dimension selected by REQUESTED as in FIT_PCA ([] = MAXIMUM, a value
% in (0, 1) = fraction of the scatter, an integer = that many), capped at
% MAXIMUM and at the number of components of the fold's PCA.
maximum = min(maximum, pcaModel.numComponents);
if isempty(requested)
    k = maximum;
elseif requested > 0 && requested < 1
    k = find(cumsum(pcaModel.explained) >= requested, 1, 'first');
    k = min(k, maximum);
else
    validateattributes(requested, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    k = min(requested, maximum);
end
end
