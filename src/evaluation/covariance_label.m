% 【檔案功能】評估：產生貝氏分類器共變異數種類的圖例文字。
%   PCA 特徵是多維的，標成 Sigma_i；LDA 特徵只有 1 維，標成 sigma_i^2。

function label = covariance_label(covarianceType, numFeatures)
%COVARIANCE_LABEL Legend text of a Bayes covariance type.
%   LABEL = COVARIANCE_LABEL(COVARIANCETYPE, NUMFEATURES) normalizes
%   COVARIANCETYPE as FIT_BAYES does ('full' or 'shared', any case or
%   abbreviation) and returns its legend text. NUMFEATURES = 1 (the LDA
%   feature) labels variances, anything else labels covariance matrices.

covarianceType = validatestring(covarianceType, {'full', 'shared'});
isFull = strcmp(covarianceType, 'full');
if numFeatures == 1
    if isFull
        label = '\sigma_i^2 separate';
    else
        label = '\sigma^2 shared';
    end
elseif isFull
    label = '\Sigma_i arbitrary (quadratic)';
else
    label = '\Sigma_i = \Sigma (linear)';
end
end
