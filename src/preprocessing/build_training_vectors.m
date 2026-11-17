% 【檔案功能】前處理：把 data/train 的人臉影像轉成特徵向量矩陣。
%   讀取 man_*.png 與 woman_*.png，保留 640x640 RGB 原始解析度，像素值縮放到
%   [0, 1]，每張影像攤平成一列，得到 X（200 x 1,228,800）與標籤 y
%   （0 = 男、1 = 女），存成 data/training_vectors_640x640x3_RGB.mat。
%   這裡不做置中或標準化，避免交叉驗證時發生資料洩漏。

function dataset = build_training_vectors(trainDir, outputFile, imageSize)
%BUILD_TRAINING_VECTORS Convert the PR training face images to feature vectors.
%   DATASET = BUILD_TRAINING_VECTORS() reads data/train/man_*.png and
%   data/train/woman_*.png, keeps each image at its original 640-by-640 RGB
%   resolution, scales its pixels to [0, 1], and saves
%   data/training_vectors_640x640x3_RGB.mat.
%
%   The saved matrix X is N-by-D: each row is one face and each column is
%   one RGB pixel-channel feature (R plane, then G plane, then B plane).
%   y uses 0 for man and 1 for woman. Do not standardize or mean-center X
%   here: PCA/LDA parameters must be fitted separately in each
%   cross-validation training fold to avoid data leakage.

if nargin < 1 || isempty(trainDir)
    paths = project_paths();
    projectDir = paths.data;
    trainDir = paths.train;
else
    projectDir = fileparts(trainDir);
end

if nargin < 3 || isempty(imageSize)
    imageSize = [];
    useOriginalSize = true;
else
    useOriginalSize = false;
end

if ~useOriginalSize
    validateattributes(imageSize, {'numeric'}, {'vector', 'numel', 2, ...
        'integer', 'positive'});
end

if ~isfolder(trainDir)
    error('Training folder not found: %s', trainDir);
end

maleFiles = dir(fullfile(trainDir, 'man_*.png'));
femaleFiles = dir(fullfile(trainDir, 'woman_*.png'));
if isempty(maleFiles) || isempty(femaleFiles)
    error('Expected both man_*.png and woman_*.png files in: %s', trainDir);
end

allFiles = [maleFiles; femaleFiles];
y = [zeros(numel(maleFiles), 1); ones(numel(femaleFiles), 1)];
labels = categorical(y, [0, 1], {'man', 'woman'});

if useOriginalSize
    firstImage = toRgbUnitRange(imread(fullfile( ...
        allFiles(1).folder, allFiles(1).name)));
    imageSize = size(firstImage, [1, 2]);
end

if nargin < 2 || isempty(outputFile)
    outputFile = fullfile(projectDir, sprintf( ...
        'training_vectors_%dx%dx3_RGB.mat', imageSize(1), imageSize(2)));
end

numImages = numel(allFiles);
numFeatures = prod(imageSize) * 3;
X = zeros(numImages, numFeatures, 'single');
fileNames = cell(numImages, 1);

for i = 1:numImages
    fullName = fullfile(allFiles(i).folder, allFiles(i).name);
    rawImage = imread(fullName);
    rgbImage = toRgbUnitRange(rawImage);
    if isequal(size(rgbImage, [1, 2]), imageSize)
        vectorImage = rgbImage;
    else
        vectorImage = zeros([imageSize, 3], 'single');
        for channel = 1:3
            vectorImage(:, :, channel) = resizeLinear( ...
                rgbImage(:, :, channel), imageSize);
        end
    end

    X(i, :) = single(vectorImage(:)).';
    fileNames{i} = allFiles(i).name;
end

if useOriginalSize
    resizeMethod = 'none (original image resolution retained)';
else
    resizeMethod = 'bilinear interpolation, applied independently to RGB channels';
end

preprocessing = struct( ...
    'colorConversion', 'none (RGB channels retained)', ...
    'resizeMethod', resizeMethod, ...
    'pixelRange', '[0, 1]', ...
    'vectorization', 'column-major flattened image', ...
    'sourceFolder', trainDir);

save(outputFile, 'X', 'y', 'labels', 'fileNames', 'imageSize', ...
    'preprocessing', '-v7');

dataset = struct('X', X, 'y', y, 'labels', labels, ...
    'fileNames', {fileNames}, 'imageSize', imageSize, ...
    'preprocessing', preprocessing);

fprintf('Created %s\n', outputFile);
fprintf('Images: %d (%d man, %d woman)\n', numImages, ...
    numel(maleFiles), numel(femaleFiles));
fprintf('Feature matrix X: %d-by-%d (%s)\n', ...
    size(X, 1), size(X, 2), class(X));
end

function rgbImage = toRgbUnitRange(rawImage)
% Keep an RGB image and scale its channel values to [0, 1].
if ndims(rawImage) ~= 3 || size(rawImage, 3) < 3
    error('Unsupported image format. Expected an RGB image with three channels.');
end
rgbImage = im2singleLocal(rawImage(:, :, 1:3));
end

function unitImage = im2singleLocal(image)
% Equivalent to im2single for the numeric image classes used by imread.
if isinteger(image)
    unitImage = single(image) ./ single(intmax(class(image)));
else
    unitImage = single(image);
    if any(unitImage(:) > 1) || any(unitImage(:) < 0)
        minValue = min(unitImage(:));
        maxValue = max(unitImage(:));
        if maxValue > minValue
            unitImage = (unitImage - minValue) ./ (maxValue - minValue);
        else
            unitImage = zeros(size(unitImage), 'single');
        end
    end
end
end

function resizedImage = resizeLinear(image, targetSize)
% Resize without requiring Image Processing Toolbox.
[sourceRows, sourceCols] = size(image);
targetRows = targetSize(1);
targetCols = targetSize(2);
[queryCols, queryRows] = meshgrid( ...
    linspace(1, sourceCols, targetCols), ...
    linspace(1, sourceRows, targetRows));
resizedImage = interp2(image, queryCols, queryRows, 'linear');
end
