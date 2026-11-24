% 【檔案功能】共變異數估計：Ledoit-Wolf 收縮（shrinkage）共變異數。
%   樣本數 n 不比維度 d 大很多時，樣本共變異數 S 的小特徵值被嚴重低估，
%   d >= n 時 S 更是奇異的。這裡把 S 往 mu * I（mu = 平均變異數）收縮：
%     Sigma = (1 - alpha) * S + alpha * mu * I
%   收縮係數 alpha 由資料本身估計，會隨 d / n 自動變大，不需要手動調整。
%   fit_bayes（類別共變異數）與 fit_fisher（類內散佈 S_W）都呼叫這個函式。
%   參考文獻：O. Ledoit and M. Wolf, "A well-conditioned estimator for
%   large-dimensional covariance matrices," Journal of Multivariate
%   Analysis, vol. 88, no. 2, pp. 365-411, 2004.

function [Sigma, alpha, S] = ledoit_wolf_covariance(Zc)
%LEDOIT_WOLF_COVARIANCE Shrunk covariance estimate of centered samples.
%   [SIGMA, ALPHA, S] = LEDOIT_WOLF_COVARIANCE(ZC) treats each row of ZC
%   (n-by-d) as one sample that has already been centered (each row minus
%   the mean of its class) and returns
%     S     = ZC' * ZC / n                    sample covariance (MLE)
%     SIGMA = (1 - ALPHA) * S + ALPHA * mu * I,  mu = trace(S) / d
%   ALPHA in [0, 1] is the Ledoit-Wolf estimate of the optimal shrinkage
%   intensity (Ledoit & Wolf 2004, Sec. 3.3):
%     delta^2 = ||S - mu I||_F^2
%     beta^2  = min(delta^2, (1/n^2) * sum_k ||z_k z_k' - S||_F^2)
%     ALPHA   = beta^2 / delta^2
%   ALPHA is close to 0 when n >> d and grows as d approaches or exceeds n,
%   so SIGMA is always positive definite, even when S is singular.

[n, d] = size(Zc);
S = (Zc.' * Zc) / n;
S = (S + S.') / 2;
mu = trace(S) / d;

delta2 = norm(S - mu * eye(d), 'fro') ^ 2;
% sum_k ||z_k z_k' - S||_F^2 = sum_k ||z_k||^4 - n ||S||_F^2
rowNorm2 = sum(Zc .^ 2, 2);
beta2 = (sum(rowNorm2 .^ 2) - n * norm(S, 'fro') ^ 2) / n ^ 2;
beta2 = min(max(beta2, 0), delta2);

if delta2 > 0
    alpha = beta2 / delta2;
else
    % S is already a multiple of the identity (always the case for d = 1).
    alpha = 0;
end
Sigma = (1 - alpha) * S + alpha * mu * eye(d);
Sigma = (Sigma + Sigma.') / 2;
end
