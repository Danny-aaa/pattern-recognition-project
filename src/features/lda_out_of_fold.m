% 【檔案功能】特徵降維：用內層交叉驗證求訓練樣本的「折外」（out-of-fold）LDA 特徵。
%   Fisher 方向是用訓練樣本求的，同一批樣本投影上去會被刻意分得很開，
%   用它們估計貝氏分類器的變異數會嚴重低估，決策門檻因此太窄。
%   這裡把訓練折再依「人」分成內層的幾折：每一內層折都用其他內層折求
%   Fisher 方向，只投影這一折的樣本。得到的特徵沒有被自己的 Fisher 方向
%   看過，分佈和真正的測試樣本相近，用來訓練貝氏分類器
%   （cross-fitting／stacking 的作法）。
%   PCA 仍然是整個訓練折一起做：PCA 不使用標籤，過度分離來自 Fisher 方向。

function zOutOfFold = lda_out_of_fold(Atrain, y, innerFoldId, dims, shrinkage)
%LDA_OUT_OF_FOLD Out-of-fold LDA features of the training samples.
%   ZOUTOFFOLD = LDA_OUT_OF_FOLD(ATRAIN, Y, INNERFOLDID, DIMS, SHRINKAGE)
%   takes the PCA components ATRAIN (n-by-P, from FIT_PCA on the training
%   samples) with labels Y and an inner fold number per sample (e.g. from
%   MAKE_PERSON_FOLDS on the training file names). For every inner fold f
%   and every p in DIMS it fits
%     w_f = FIT_FISHER(ATRAIN(innerFoldId ~= f, 1:p), y(...), SHRINKAGE)
%   and sets ZOUTOFFOLD(innerFoldId == f, j) = ATRAIN(innerFoldId == f, 1:p) * w_f.
%   ZOUTOFFOLD is n-by-numel(DIMS); fit the Bayes classifier on its column
%   for p and apply it to the test features projected with the Fisher
%   direction of all training samples.
%
%   Each inner training set has fewer samples than the whole training fold,
%   so S_W of the inner fit is singular for p > n_inner - 2. That case
%   requires SHRINKAGE = 'ledoit-wolf' (the default of FIT_FISHER).

if nargin < 5
    shrinkage = [];
end
y = y(:);
innerFoldId = innerFoldId(:);
if numel(innerFoldId) ~= size(Atrain, 1) || numel(y) ~= size(Atrain, 1)
    error('Atrain, y and innerFoldId must have the same number of samples.');
end
if max(dims) > size(Atrain, 2)
    error('Atrain has %d components; p = %d was requested.', ...
        size(Atrain, 2), max(dims));
end

folds = unique(innerFoldId).';
if numel(folds) < 2
    error('At least two inner folds are required.');
end
smallestInner = min(arrayfun(@(f) nnz(innerFoldId ~= f), folds));
usesShrinkage = isempty(shrinkage) || ...
    strcmp(validatestring(shrinkage, {'ledoit-wolf', 'none'}), 'ledoit-wolf');
if ~usesShrinkage && max(dims) > smallestInner - 2
    error(['p = %d exceeds n_inner - 2 = %d, where the inner S_W is ' ...
        'singular. Use Ledoit-Wolf shrinkage or smaller p.'], ...
        max(dims), smallestInner - 2);
end

zOutOfFold = zeros(size(Atrain, 1), numel(dims));
for f = folds
    isHeldOut = (innerFoldId == f);
    for j = 1:numel(dims)
        p = dims(j);
        fisher = fit_fisher(Atrain(~isHeldOut, 1:p), y(~isHeldOut), shrinkage);
        zOutOfFold(isHeldOut, j) = Atrain(isHeldOut, 1:p) * fisher.w;
    end
end
end
