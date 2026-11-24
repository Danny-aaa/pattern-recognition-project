% 【檔案功能】特徵降維：用 fit_pca 或 fit_lda 得到的模型把資料投影到低維特徵空間。
%   計算 Z = (X - 訓練平均) * W，每一列是一張影像降維後的特徵向量。
%   影像與模型以 single 儲存以節省記憶體，但 120 萬維的內積若用 single 累加，
%   誤差約 1e-5 到 1e-4，會汙染小特徵值方向；因此一律轉成 double 計算，
%   並分批處理資料列以限制記憶體用量。

function Z = project_features(model, X)
%PROJECT_FEATURES Project the rows of X with a model from FIT_PCA or FIT_LDA.
%   Z = PROJECT_FEATURES(MODEL, X) returns Z = (X - MODEL.mean) * MODEL.W,
%   one projected feature vector per row of X. This is
%     PCA: a_k = E' (x_k - m)           PR_ch3 Eq. (53)
%     LDA: y_k = w' x_k (shifted by m)  PR_ch3 p. 105
%   Subtracting the training mean only shifts the LDA output by a constant
%   and does not change the class separation.
%
%   The products are accumulated in double precision (as in FIT_PCA):
%   summing 1.2 million single-precision products is not accurate enough
%   for the components with small eigenvalues. Rows are processed in
%   blocks, so only one block of X is held in double at a time.

blockSize = 20;
W = double(model.W);
m = double(model.mean);
n = size(X, 1);
Z = zeros(n, size(W, 2));
for first = 1:blockSize:n
    rows = first:min(first + blockSize - 1, n);
    Z(rows, :) = (double(X(rows, :)) - m) * W;
end
end
