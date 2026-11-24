% 【檔案功能】特徵降維：對男／女兩類資料做 Fisher 線性鑑別分析（LDA）。
%   類內散佈矩陣 S_W 在樣本數少於維度時是奇異的，因此先用 PCA 降維，再在
%   PCA 子空間內呼叫 fit_fisher 求 Fisher 方向 w = S_W^-1 (c1 - c2)
%   （Fisherface 作法）。預設對 S_W 做 Ledoit-Wolf 收縮（見 fit_fisher）。
%   兩類問題的 LDA 輸出是 1 維特徵（c - 1 = 1）。
%   要接貝氏分類器時，請給 innerFoldId 並用第三個輸出（折外 LDA 特徵）訓練
%   貝氏分類器；第二個輸出是被 Fisher 方向看過的訓練投影，變異數會被低估。

function [model, ztrain, zOutOfFold] = fit_lda(X, y, pcaComponents, ...
    shrinkage, innerFoldId)
%FIT_LDA Fit Fisher's linear discriminant for the man/woman labels.
%   MODEL = FIT_LDA(X, Y, PCACOMPONENTS) treats each row of X (n-by-d) as
%   one sample with label Y (0 = man, 1 = woman) and returns the Fisher
%   direction w. Project samples with PROJECT_FEATURES(MODEL, Xnew); larger
%   projections indicate the woman class. For two classes LDA gives a
%   one-dimensional feature (c - 1 = 1).
%
%   The within-class scatter S_W is d-by-d with rank at most n - 2, so it
%   is singular whenever n <= d (PR_ch3 p. 84) and w* = S_W^-1 (c1 - c2)
%   cannot be applied to the 1,228,800-dimensional pixels directly. The
%   data are therefore first reduced with FIT_PCA (the "Fisherface"
%   approach) and Fisher's criterion is solved in that subspace.
%   PCACOMPONENTS is passed to FIT_PCA; the default n - 2 is the largest
%   dimension in which S_W is nonsingular.
%   SHRINKAGE is passed to FIT_FISHER: 'ledoit-wolf' (default) or 'none'.
%   ZTRAIN is the one-dimensional LDA feature of the rows of X, computed
%   from the PCA components without projecting X a second time.
%   [MODEL, ZTRAIN, ZOUTOFFOLD] = FIT_LDA(X, Y, PCACOMPONENTS, SHRINKAGE,
%   INNERFOLDID) also returns the out-of-fold LDA features of LDA_OUT_OF_FOLD
%   for the inner folds INNERFOLDID (e.g. MAKE_PERSON_FOLDS on the file
%   names of X). Fit the Bayes classifier on ZOUTOFFOLD, not on ZTRAIN:
%   the Fisher direction over-separates the samples it was fitted on.
%
%   Formulas (PR_ch3, Sec. 3.8.2), with class 1 = woman and class 2 = man:
%     c_i = (1/N_i) * sum_{x in class i} x                       p. 109
%     S_i = sum_{x in class i} (x - c_i)(x - c_i)'               p. 113
%     S_W = S_1 + S_2                                            p. 115
%     S_B = (c_1 - c_2)(c_1 - c_2)'                              p. 116
%     J(w) = (w' S_B w) / (w' S_W w)                             p. 117
%     w* = S_W^-1 (c_1 - c_2)                                    p. 119
%   The direction in pixel space is w = E * w_pca, so that
%   w' (x - m) = w_pca' * E' (x - m).
%   Fit this only on the training fold to avoid data leakage.

y = y(:);
if size(X, 1) ~= numel(y)
    error('The number of rows in X must match the number of labels in y.');
end

n = numel(y);
if nargin < 3 || isempty(pcaComponents)
    pcaComponents = n - 2;
end
if nargin < 4
    shrinkage = [];
end

[pcaModel, Atrain] = fit_pca(X, pcaComponents);
[model, ztrain] = lda_from_pca(pcaModel, Atrain, y, ...
    pcaModel.numComponents, shrinkage);
model.pcaModel = pcaModel;

if nargout > 2
    if nargin < 5 || isempty(innerFoldId)
        error('innerFoldId is required for the out-of-fold LDA features.');
    end
    zOutOfFold = lda_out_of_fold(Atrain, y, innerFoldId, ...
        pcaModel.numComponents, shrinkage);
end
end
