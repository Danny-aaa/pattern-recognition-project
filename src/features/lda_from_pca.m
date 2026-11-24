% 【檔案功能】特徵降維：在已經算好的 PCA 子空間內建立 LDA（Fisherface）模型。
%   取 PCA 的前 p 個主成分，呼叫 fit_fisher 求 Fisher 方向，再把方向換回
%   像素空間 w = E * w_pca，得到可以直接投影新影像的 LDA 模型。
%   fit_lda 與 run_cross_validation 共用這個函式：交叉驗證的每一折只做一次
%   PCA，PCA + Bayes 與 LDA + Bayes 都從同一組主成分取用。

function [model, ztrain] = lda_from_pca(pcaModel, Atrain, y, p, shrinkage)
%LDA_FROM_PCA LDA model built on the first p components of a PCA model.
%   [MODEL, ZTRAIN] = LDA_FROM_PCA(PCAMODEL, ATRAIN, Y, P, SHRINKAGE) uses
%   the PCA model from FIT_PCA and its training components ATRAIN (the
%   second output of FIT_PCA), keeps the first P components, and solves
%   Fisher's criterion there with FIT_FISHER(ATRAIN(:, 1:P), Y, SHRINKAGE).
%   MODEL is used with PROJECT_FEATURES like the model of FIT_LDA, and
%   ZTRAIN = ATRAIN(:, 1:P) * w is the one-dimensional LDA feature of the
%   training samples. SHRINKAGE defaults to 'ledoit-wolf'.

if nargin < 5
    shrinkage = [];
end
if p > pcaModel.numComponents
    error('The PCA model has %d components; p = %d was requested.', ...
        pcaModel.numComponents, p);
end

y = y(:);
Ap = Atrain(:, 1:p);
fisher = fit_fisher(Ap, y, shrinkage);
if strcmp(fisher.shrinkageType, 'none') && fisher.rcondSw < 1e-12
    warning(['S_W is close to singular (rcond = %.2e). Use fewer PCA ' ...
        'components or Ledoit-Wolf shrinkage.'], fisher.rcondSw);
end
wPca = fisher.w;
ztrain = Ap * wPca;

model = struct( ...
    'method', 'LDA', ...
    'mean', pcaModel.mean, ...
    'W', single(double(pcaModel.W(:, 1:p)) * wPca), ...
    'wPca', wPca, ...
    'Sw', fisher.Sw, ...
    'Sb', fisher.Sb, ...
    'fisherCriterion', fisher.fisherCriterion, ...
    'shrinkageType', fisher.shrinkageType, ...
    'shrinkage', fisher.shrinkage, ...
    'numComponents', 1, ...
    'numPcaComponents', p, ...
    'numTrainingSamples', numel(y));
end
