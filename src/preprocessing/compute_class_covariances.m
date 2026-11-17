% 【檔案功能】前處理：計算男性與女性影像各自的平均向量與共變異數矩陣（第四、五週）。
%   原始的共變異數矩陣是 1,228,800 x 1,228,800，無法直接儲存，因此改存
%   樣本空間的精簡表示 Gk = Xc * Xc' / (Nk - 1)（Nk x Nk），它和原矩陣有相同的
%   非零特徵值。結果存成 data/class_covariances_compact_RGB.mat，
%   內含 Gman、Gwoman、兩類的平均向量與平均臉。

function result = compute_class_covariances(datasetFile, outputFile, sampleIdx)
%COMPUTE_CLASS_COVARIANCES Compute compact covariance representations for men and women.
%   RESULT = COMPUTE_CLASS_COVARIANCES() loads the RGB training-vector data,
%   calculates the mean vector of each class, and calculates the
%   sample-space covariance (Gram) matrices Gman and Gwoman.
%
%   RESULT = COMPUTE_CLASS_COVARIANCES(DATASETFILE, OUTPUTFILE, SAMPLEIDX)
%   uses only the rows of X selected by SAMPLEIDX (logical or index
%   vector). Use this inside 5-fold cross-validation so that the
%   statistics are estimated from the training fold only. Pass '' as
%   OUTPUTFILE to skip saving.
%
%   The feature-space covariance of class k, Ck = Xc' * Xc / (Nk - 1),
%   would be 1,228,800-by-1,228,800 for the original RGB images. It is
%   therefore represented compactly by Gk = Xc * Xc' / (Nk - 1), which is
%   Nk-by-Nk and has the same nonzero eigenvalues as Ck. An eigenvector v
%   of Gk maps to an eigenvector of Ck as u = Xc' * v / norm(Xc' * v).

paths = project_paths();
if nargin < 1 || isempty(datasetFile)
    datasetFile = fullfile(paths.data, ...
        'training_vectors_640x640x3_RGB.mat');
end
if nargin < 2
    outputFile = fullfile(paths.data, 'class_covariances_compact_RGB.mat');
end

if ~isfile(datasetFile)
    error('Training-vector file not found: %s', datasetFile);
end

data = load(datasetFile, 'X', 'y', 'fileNames', 'imageSize');
if size(data.X, 1) ~= numel(data.y)
    error('The number of rows in X must match the number of labels in y.');
end

numSamples = numel(data.y);
if nargin < 3 || isempty(sampleIdx)
    useRow = true(numSamples, 1);
else
    useRow = false(numSamples, 1);
    useRow(sampleIdx) = true;
end

imageSize = data.imageSize;
numFeatures = size(data.X, 2);
numChannels = numFeatures / prod(imageSize);
if numChannels ~= 3
    error('Expected RGB vectors with three channels; found %g.', numChannels);
end

man = classStatistics(data, useRow & (data.y == 0), 'male', imageSize);
woman = classStatistics(data, useRow & (data.y == 1), 'female', imageSize);
clear data

Gman = man.G;
Gwoman = woman.G;
manMeanVector = man.meanVector;
womanMeanVector = woman.meanVector;
meanManFace = man.meanFace;
meanWomanFace = woman.meanFace;
maleFileNames = man.fileNames;
femaleFileNames = woman.fileNames;

covarianceInfo = struct( ...
    'representation', 'Sample-space covariance (Gram) matrix per class', ...
    'formula', 'Gk = Xc * Xc'' / (Nk - 1), Xc = Xk - mean(Xk)', ...
    'fullCovarianceFormula', 'Ck = Xc'' * Xc / (Nk - 1)', ...
    'fullCovarianceSize', [numFeatures, numFeatures], ...
    'numMaleImages', size(Gman, 1), ...
    'numFemaleImages', size(Gwoman, 1), ...
    'traceCman', trace(Gman), ...
    'traceCwoman', trace(Gwoman), ...
    'numFeatures', numFeatures, ...
    'imageSize', imageSize, ...
    'numChannels', numChannels);

% The centered data matrices are intentionally not saved: each is about
% 469 MiB and can be regenerated from the source data.
if ~isempty(outputFile)
    save(outputFile, 'Gman', 'Gwoman', 'manMeanVector', ...
        'womanMeanVector', 'meanManFace', 'meanWomanFace', ...
        'maleFileNames', 'femaleFileNames', 'covarianceInfo', '-v7');
    fprintf('Created %s\n', outputFile);
end

result = struct('Gman', Gman, 'Gwoman', Gwoman, ...
    'manMeanVector', manMeanVector, 'womanMeanVector', womanMeanVector, ...
    'meanManFace', meanManFace, 'meanWomanFace', meanWomanFace, ...
    'maleFileNames', {maleFileNames}, 'femaleFileNames', {femaleFileNames}, ...
    'covarianceInfo', covarianceInfo);

fprintf('Male images: %d, female images: %d\n', size(Gman, 1), size(Gwoman, 1));
fprintf('Mean vectors: 1-by-%d\n', numFeatures);
fprintf('Compact covariance Gman: %d-by-%d, trace = %.4f\n', ...
    size(Gman, 1), size(Gman, 2), trace(Gman));
fprintf('Compact covariance Gwoman: %d-by-%d, trace = %.4f\n', ...
    size(Gwoman, 1), size(Gwoman, 2), trace(Gwoman));
end

function stats = classStatistics(data, classIdx, className, imageSize)
% Mean vector, mean face, and compact covariance for one class.
numClass = nnz(classIdx);
if numClass < 2
    error('At least two %s images are required to calculate covariance.', ...
        className);
end

% Work in double: summing 1.2 million single-precision products gives
% errors large enough to break the expected rank Nk - 1 of Gk.
Xc = double(data.X(classIdx, :));
meanVector = mean(Xc, 1);
Xc = Xc - meanVector;
meanVector = single(meanVector);

% Symmetrize to remove tiny floating-point asymmetries before an
% eigendecomposition.
G = (Xc * Xc.') / (numClass - 1);
G = (G + G.') / 2;

stats = struct('G', G, 'meanVector', meanVector, ...
    'meanFace', reshape(meanVector, [imageSize, 3]), ...
    'fileNames', {data.fileNames(classIdx)});
end
