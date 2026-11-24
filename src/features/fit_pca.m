% 【檔案功能】特徵降維：對訓練資料做主成分分析（PCA），求出投影用的特徵向量。
%   scatter matrix S 是 d x d（d = 1,228,800）無法直接建立，因此改解 n x n 的
%   Xc * Xc' 的特徵值問題，再換算回 S 的特徵向量（Turk & Pentland 的
%   eigenfaces 作法）。資料置中後最多只有 n - 1 個非零特徵值，所以 PCA 的
%   維度上限是 n - 1。可用個數或解釋變異比例指定要保留的維度。
%   第二個輸出直接給出訓練資料的 PCA 特徵 V * sqrt(lambda)，等於
%   (X - m) * E，但不必再對 120 萬維的資料做一次投影。

function [model, Atrain] = fit_pca(X, numComponents)
%FIT_PCA Fit principal component analysis to the rows of X.
%   MODEL = FIT_PCA(X, NUMCOMPONENTS) treats each row of X (n-by-d) as one
%   sample and returns the d'-dimensional PCA subspace of the training
%   data. Project new samples with PROJECT_FEATURES(MODEL, Xnew), which
%   computes the principal components a = E' * (x - m).
%
%   [MODEL, ATRAIN] = FIT_PCA(...) also returns the n-by-d' principal
%   components of the training samples themselves. Since
%   Xc * E = Xc * Xc' * V / sqrt(lambda) = V * sqrt(lambda), they come from
%   the n-by-n eigenproblem directly, in double precision, and equal
%   PROJECT_FEATURES(MODEL, X) without a second pass over the pixels.
%
%   NUMCOMPONENTS selects d':
%     integer >= 1    keep that many components (capped at rank n - 1)
%     value in (0,1)  keep the fewest components whose eigenvalues explain
%                     at least that fraction of the total scatter
%     [] (default)    keep all n - 1 nonzero components
%
%   Formulas (PR_ch3, Sec. 3.8.1):
%     m = (1/n) * sum_k x_k                          Eq. (50)
%     S = sum_k (x_k - m)(x_k - m)'                  Eq. (54)
%     S e = lambda e, keep the d' largest lambda     Eqs. (58), (60)
%     a_k = e' (x_k - m)                             Eq. (53)
%
%   S is d-by-d (d = 1,228,800 for the RGB faces), so it is never formed.
%   With Xc the centered data, S = Xc' * Xc and its nonzero eigenpairs
%   follow from the n-by-n matrix Xc * Xc':
%     (Xc * Xc') v = lambda v  =>  S (Xc' v) = lambda (Xc' v),
%   so e = Xc' * v / sqrt(lambda) is a unit eigenvector of S.
%   Fit this only on the training fold to avoid data leakage.

if nargin < 2
    numComponents = [];
end

n = size(X, 1);
if n < 2
    error('At least two samples are required for PCA.');
end

% Work in double: summing 1.2 million single-precision products is not
% accurate enough for the small eigenvalues.
Xc = double(X);
m = mean(Xc, 1);
Xc = Xc - m;

G = Xc * Xc.';
G = (G + G.') / 2;
[V, D] = eig(G);
[lambda, order] = sort(diag(D), 'descend');
V = V(:, order);

% Centering leaves at most n - 1 nonzero eigenvalues; drop the numerically
% zero ones before dividing by sqrt(lambda).
tolerance = max(lambda) * n * eps;
maxComponents = min(nnz(lambda > tolerance), n - 1);
explained = lambda(1:maxComponents) / sum(lambda(1:maxComponents));

if isempty(numComponents)
    k = maxComponents;
elseif numComponents > 0 && numComponents < 1
    k = find(cumsum(explained) >= numComponents, 1, 'first');
else
    validateattributes(numComponents, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    k = min(numComponents, maxComponents);
end

E = (Xc.' * V(:, 1:k)) ./ sqrt(lambda(1:k)).';
if nargout > 1
    Atrain = V(:, 1:k) .* sqrt(lambda(1:k)).';
end

model = struct( ...
    'method', 'PCA', ...
    'mean', single(m), ...
    'W', single(E), ...
    'eigenvalues', lambda(1:k), ...
    'explained', explained(1:k), ...
    'numComponents', k, ...
    'numTrainingSamples', n);
end
