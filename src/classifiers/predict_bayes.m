% 【檔案功能】分類器：用 fit_bayes 訓練好的模型預測性別。
%   計算兩類的鑑別函數 g_i(x)，輸出預測標籤（0 = 男、1 = 女）、
%   分數 g_女(x) - g_男(x)（大於 0 判為女，也用來畫 ROC 曲線）
%   以及事後機率 P(女 | x)。

function [yPred, score, posteriorWoman] = predict_bayes(model, Z)
%PREDICT_BAYES Classify the rows of Z with a model from FIT_BAYES.
%   [YPRED, SCORE, POSTERIORWOMAN] = PREDICT_BAYES(MODEL, Z) returns the
%   predicted labels (0 = man, 1 = woman), the discriminant
%   SCORE = g_woman(x) - g_man(x), and the posterior P(woman | x).
%   SCORE > 0 means woman; use SCORE for the ROC curve.
%
%   Formulas (PR_ch2):
%     g_i(x) = -1/2 (x - mu_i)' Sigma_i^-1 (x - mu_i) - d/2 ln(2 pi)
%              - 1/2 ln|Sigma_i| + ln P(w_i)              Eq. (41), p. 94
%     g(x)   = g_1(x) - g_2(x)
%            = ln p(x|w_1)/p(x|w_2) + ln P(w_1)/P(w_2)    Eq. (27), p. 75
%     decide w_1 if g(x) > 0, with w_1 = woman and w_2 = man
%     P(w_1|x) = p(x|w_1) P(w_1) / p(x) = 1 / (1 + exp(-g(x)))
%                                                         Eqs. (2)-(3), p. 11

d = size(Z, 2);
if d ~= model.numFeatures
    error('Z has %d features; the model expects %d.', d, model.numFeatures);
end

g = zeros(size(Z, 1), 2);
for i = 1:2
    % Cholesky factor: Sigma = R' R, so the Mahalanobis term is
    % ||(x - mu) / R||^2 and ln|Sigma| = 2 * sum(ln(diag(R))).
    R = chol(model.Sigma(:, :, i));
    U = (Z - model.mu(i, :)) / R;
    g(:, i) = -0.5 * sum(U .^ 2, 2) - d / 2 * log(2 * pi) ...
        - sum(log(diag(R))) + log(model.prior(i));
end

score = g(:, 2) - g(:, 1);
yPred = double(score > 0);
posteriorWoman = 1 ./ (1 + exp(-score));
end
