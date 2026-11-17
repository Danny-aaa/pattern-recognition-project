% 【檔案功能】評估：對 PCA + Bayes 與 LDA + Bayes 進行依人分組的五折交叉驗證。
%   每一折都只用訓練折來訓練降維與分類器，再套用到測試折。輸出各方法的
%   準確率、AUC、特徵維度、參數量、訓練時間與推論時間，並把合併後的
%   ROC 曲線存成 results/cv_roc_curves.png、數值存成 results/cv_results.mat。

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
%   OPTIONS fields (all optional):
%     numFolds          5
%     seed              1      fold assignment seed
%     pcaComponents     20     d' for PCA + Bayes (see FIT_PCA)
%     ldaPcaComponents  50     PCA dimension inside LDA. The Fisherface
%                              maximum n - 2 shrinks the projected
%                              within-class scatter to almost zero on the
%                              training folds and the Bayes threshold
%                              then misplaces some test faces.
%     covarianceType    'full' Bayes covariance (see FIT_BAYES)
%     regularization    1e-6   Bayes covariance regularization
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
    'ldaPcaComponents', 50, 'covarianceType', 'full', ...
    'regularization', 1e-6, ...
    'plotFile', fullfile(paths.results, 'cv_roc_curves.png'), ...
    'resultFile', fullfile(paths.results, 'cv_results.mat'), 'verbose', true);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(options, names{i})
        options.(names{i}) = defaults.(names{i});
    end
end

if ~isfile(datasetFile)
    error('Training-vector file not found: %s', datasetFile);
end
data = load(datasetFile, 'X', 'y', 'fileNames');
X = data.X;
y = data.y(:);
foldId = make_person_folds(data.fileNames, y, options.numFolds, options.seed);
clear data

methods = struct( ...
    'name', {'PCA + Bayes', 'LDA + Bayes'}, ...
    'fitReducer', { ...
        @(Xtr, ytr) fit_pca(Xtr, options.pcaComponents), ...
        @(Xtr, ytr) fit_lda(Xtr, ytr, options.ldaPcaComponents)});

numMethods = numel(methods);
numFolds = options.numFolds;
accuracy = zeros(numFolds, numMethods);
auc = zeros(numFolds, numMethods);
trainTime = zeros(numFolds, numMethods);
inferenceTime = zeros(numFolds, numMethods);
featureDim = zeros(numFolds, numMethods);
projectionParams = zeros(numFolds, numMethods);
classifierParams = zeros(numFolds, numMethods);
scores = zeros(numel(y), numMethods);

for k = 1:numFolds
    isTest = (foldId == k);
    Xtr = X(~isTest, :);
    ytr = y(~isTest);
    Xte = X(isTest, :);
    yte = y(isTest);

    for m = 1:numMethods
        tic;
        reducer = methods(m).fitReducer(Xtr, ytr);
        classifier = fit_bayes(project_features(reducer, Xtr), ytr, ...
            options.covarianceType, options.regularization);
        trainTime(k, m) = toc;

        tic;
        [yPred, score] = predict_bayes(classifier, ...
            project_features(reducer, Xte));
        inferenceTime(k, m) = toc / numel(yte);

        scores(isTest, m) = score;
        accuracy(k, m) = mean(yPred == yte);
        [~, ~, auc(k, m)] = roc_auc(score, yte);
        featureDim(k, m) = size(reducer.W, 2);
        projectionParams(k, m) = numel(reducer.W) + numel(reducer.mean);
        classifierParams(k, m) = classifier.numParameters;

        if options.verbose
            fprintf('Fold %d/%d  %-12s acc = %.3f  AUC = %.4f  d'' = %d\n', ...
                k, numFolds, methods(m).name, accuracy(k, m), auc(k, m), ...
                featureDim(k, m));
        end
    end
    clear Xtr Xte
end

roc = struct('falseAlarm', cell(1, numMethods), 'hit', [], 'auc', []);
for m = 1:numMethods
    [roc(m).falseAlarm, roc(m).hit, roc(m).auc] = roc_auc(scores(:, m), y);
end

results = table( ...
    string({methods.name}).', ...
    mean(featureDim, 1).', ...
    mean(accuracy, 1).', std(accuracy, 0, 1).', ...
    mean(auc, 1).', [roc.auc].', ...
    mean(projectionParams, 1).', mean(classifierParams, 1).', ...
    mean(trainTime, 1).', 1000 * mean(inferenceTime, 1).', ...
    'VariableNames', {'Method', 'FeatureDim', 'AccuracyMean', ...
    'AccuracyStd', 'AUCMean', 'AUCPooled', 'ProjectionParams', ...
    'ClassifierParams', 'TrainTime_s', 'InferenceTime_ms'});

details = struct('options', options, 'foldId', foldId, ...
    'methodNames', {{methods.name}}, 'accuracy', accuracy, 'auc', auc, ...
    'trainTime', trainTime, 'inferenceTime', inferenceTime, ...
    'featureDim', featureDim, 'projectionParams', projectionParams, ...
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
            'DisplayName', sprintf('%s (AUC = %.4f)', methods(m).name, ...
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
