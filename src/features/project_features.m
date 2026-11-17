% 【檔案功能】特徵降維：用 fit_pca 或 fit_lda 得到的模型把資料投影到低維特徵空間。
%   計算 Z = (X - 訓練平均) * W，每一列是一張影像降維後的特徵向量。

function Z = project_features(model, X)
%PROJECT_FEATURES Project the rows of X with a model from FIT_PCA or FIT_LDA.
%   Z = PROJECT_FEATURES(MODEL, X) returns Z = (X - MODEL.mean) * MODEL.W,
%   one projected feature vector per row of X. This is
%     PCA: a_k = E' (x_k - m)           PR_ch3 Eq. (53)
%     LDA: y_k = w' x_k (shifted by m)  PR_ch3 p. 105
%   Subtracting the training mean only shifts the LDA output by a constant
%   and does not change the class separation.

Z = double((X - model.mean) * model.W);
end
