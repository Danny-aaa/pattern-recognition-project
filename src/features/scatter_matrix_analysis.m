% 【檔案功能】第九週進度：scatter matrix 的記憶體需求、特徵向量與 PCA 維度上限。
%   1. 計算儲存 scatter matrix S（d x d，d = 1,228,800）所需的記憶體空間，
%      並和實際採用的 n x n Gram 矩陣作法比較。
%   2. 不建立 S，透過 fit_pca 求出 S 的特徵向量，並驗證 S e = lambda e 與正交性。
%   3. 印出理論上 PCA 的維度上限 min(n - 1, d)，並和實際求得的非零特徵值個數對照。
%   輸出 results/scatter_eigenvectors.mat、eigenfaces.png、scatter_eigenvalues.png。
%   參考文獻：M. Turk and A. Pentland, "Eigenfaces for recognition,"
%   Journal of Cognitive Neuroscience, vol. 3, no. 1, pp. 71-86, 1991.

function result = scatter_matrix_analysis(datasetFile, numEigenfaces)
%SCATTER_MATRIX_ANALYSIS Memory cost and eigenvectors of the scatter matrix.
%   RESULT = SCATTER_MATRIX_ANALYSIS() uses all training faces to
%     1. report the memory needed to store the scatter matrix
%          S = sum_k (x_k - m)(x_k - m)'          PR_ch3 Eq. (54)
%        and compare it with what is actually computed;
%     2. compute the eigenvectors of S, S e = lambda e (PR_ch3 Eq. (58)),
%        without forming S, using FIT_PCA (eigenvectors of Xc * Xc');
%     3. check S e = lambda e as S e = Xc' * (Xc * e), and save the
%        eigenvalues, the leading eigenvectors, and an eigenface figure.
%
%   NUMEIGENFACES (default 20) is the number of leading eigenvectors saved
%   and shown. All n - 1 eigenvalues are saved.

paths = project_paths();
if nargin < 1 || isempty(datasetFile)
    datasetFile = fullfile(paths.data, 'training_vectors_640x640x3_RGB.mat');
end
if nargin < 2 || isempty(numEigenfaces)
    numEigenfaces = 20;
end

data = load(datasetFile, 'X', 'imageSize');
X = data.X;
imageSize = data.imageSize;
clear data
[n, d] = size(X);

% 1. Memory -------------------------------------------------------------
bytesDouble = 8;
bytesSingle = 4;
memory = struct( ...
    'd', d, ...
    'n', n, ...
    'numEntriesS', d^2, ...
    'S_double_bytes', d^2 * bytesDouble, ...
    'S_single_bytes', d^2 * bytesSingle, ...
    'S_upperTriangle_double_bytes', d * (d + 1) / 2 * bytesDouble, ...
    'gram_double_bytes', n^2 * bytesDouble, ...
    'X_single_bytes', n * d * bytesSingle, ...
    'eigenvectors_double_bytes', d * (n - 1) * bytesDouble);

fprintf('Scatter matrix S is %d-by-%d (%.4g entries)\n', d, d, d^2);
fprintf('  S in double            : %s\n', formatBytes(memory.S_double_bytes));
fprintf('  S in single            : %s\n', formatBytes(memory.S_single_bytes));
fprintf('  S upper triangle, double: %s\n', ...
    formatBytes(memory.S_upperTriangle_double_bytes));
fprintf('Actually computed instead:\n');
fprintf('  Gram matrix Xc*Xc'' (%dx%d, double): %s\n', n, n, ...
    formatBytes(memory.gram_double_bytes));
fprintf('  Data matrix X (%dx%d, single)    : %s\n', n, d, ...
    formatBytes(memory.X_single_bytes));
fprintf('  All %d eigenvectors (double)       : %s\n', n - 1, ...
    formatBytes(memory.eigenvectors_double_bytes));

% 2. Eigenvectors -------------------------------------------------------
tic;
pcaModel = fit_pca(X);
eigTime = toc;
E = double(pcaModel.W);
lambda = pcaModel.eigenvalues;
fprintf('Computed %d eigenvectors of S in %.1f s\n', size(E, 2), eigTime);

% PCA dimension limit: S = Xc' * Xc has rank(Xc) nonzero eigenvalues, and
% the n centered rows of Xc sum to zero, so rank(Xc) <= min(n - 1, d).
pcaDimensionLimit = min(n - 1, d);
fprintf('PCA dimension upper bound min(n - 1, d) = %d; nonzero eigenvalues found: %d\n', ...
    pcaDimensionLimit, numel(lambda));

% 3. Check S e = lambda e and orthonormality ----------------------------
Xc = double(X) - double(pcaModel.mean);
check = min(5, size(E, 2));
residual = zeros(check, 1);
for i = 1:check
    Se = Xc.' * (Xc * E(:, i));
    residual(i) = norm(Se - lambda(i) * E(:, i)) / lambda(i);
end
clear Xc
orthoError = max(abs(E.' * E - eye(size(E, 2))), [], 'all');
fprintf('max ||S e - lambda e|| / lambda (first %d): %.2e\n', check, ...
    max(residual));
fprintf('max |E''E - I|: %.2e\n', orthoError);
explained = cumsum(pcaModel.explained);
for target = [0.80, 0.90, 0.95, 0.99]
    fprintf('  %2.0f%% of the scatter: %d eigenvectors\n', 100 * target, ...
        find(explained >= target, 1));
end

k = min(numEigenfaces, size(E, 2));
eigenvectors = single(E(:, 1:k));
eigenvalues = lambda;
explainedRatio = pcaModel.explained;
meanVector = pcaModel.mean;
save(fullfile(paths.results, 'scatter_eigenvectors.mat'), 'eigenvalues', ...
    'explainedRatio', 'eigenvectors', 'meanVector', 'imageSize', ...
    'memory', 'pcaDimensionLimit', '-v7');

plotEigenfaces(E(:, 1:k), meanVector, lambda, imageSize, ...
    fullfile(paths.results, 'eigenfaces.png'));
plotSpectrum(pcaModel.explained, n, fullfile(paths.results, ...
    'scatter_eigenvalues.png'));
fprintf('Saved results/scatter_eigenvectors.mat, eigenfaces.png, scatter_eigenvalues.png\n');

result = struct('memory', memory, 'eigenvalues', eigenvalues, ...
    'explainedRatio', explainedRatio, 'eigenvectors', eigenvectors, ...
    'pcaDimensionLimit', pcaDimensionLimit, ...
    'residual', residual, 'orthoError', orthoError, 'eigTime', eigTime);
end

function text = formatBytes(bytes)
units = {'B', 'KB', 'MB', 'GB', 'TB'};
power = min(floor(log10(max(bytes, 1)) / 3), numel(units) - 1);
text = sprintf('%.2f %s (%.4g bytes)', bytes / 1000^power, units{power + 1}, ...
    bytes);
end

function plotEigenfaces(E, meanVector, lambda, imageSize, fileName)
% Mean face followed by the leading eigenvectors, each rescaled to [0, 1].
k = size(E, 2);
numCols = 7;
numRows = ceil((k + 1) / numCols);
fig = figure('Visible', 'off', 'Position', [0, 0, 1400, 220 * numRows]);
if exist('theme', 'file')
    theme(fig, 'light');
end
layout = tiledlayout(fig, numRows, numCols, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
nexttile(layout);
imshowLocal(reshape(double(meanVector), [imageSize, 3]));
title('Mean face m');
for i = 1:k
    nexttile(layout);
    face = reshape(E(:, i), [imageSize, 3]);
    imshowLocal((face - min(face(:))) / (max(face(:)) - min(face(:))));
    title(sprintf('e_{%d}, \\lambda = %.0f', i, lambda(i)));
end
exportgraphics(fig, fileName, 'Resolution', 100);
close(fig);
end

function imshowLocal(rgb)
% Show an RGB image in [0, 1] without Image Processing Toolbox.
image(min(max(rgb(1:4:end, 1:4:end, :), 0), 1));
axis image off;
end

function plotSpectrum(explained, n, fileName)
fig = figure('Visible', 'off', 'Position', [0, 0, 900, 380]);
if exist('theme', 'file')
    theme(fig, 'light');
end
yyaxis left;
bar(100 * explained);
ylabel('Eigenvalue share of total scatter (%)');
yyaxis right;
plot(100 * cumsum(explained), 'LineWidth', 1.5);
ylabel('Cumulative (%)');
ylim([0, 100]);
xlabel('Eigenvector index i (sorted by \lambda_i)');
title(sprintf('Eigenvalues of the scatter matrix S (all %d training faces)', n));
grid on;
exportgraphics(fig, fileName, 'Resolution', 150);
close(fig);
end
