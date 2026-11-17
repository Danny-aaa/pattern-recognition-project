% 【檔案功能】分類器：訓練高斯貝氏分類器（第六週）。
%   用降維後的特徵估計男、女兩類各自的平均向量、共變異數矩陣與事前機率。
%   共變異數可選各類獨立（'full'，二次決策邊界）或兩類共用（'shared'，
%   線性決策邊界），並在對角線加上少量正則化避免矩陣奇異。

function model = fit_bayes(Z, y, covarianceType, regularization)
%FIT_BAYES Fit a Gaussian Bayes classifier for the man/woman labels.
%   MODEL = FIT_BAYES(Z, Y) treats each row of Z (n-by-d', the PCA or LDA
%   features) as one sample with label Y (0 = man, 1 = woman) and models
%   each class as a multivariate normal density. Classify with
%   PREDICT_BAYES(MODEL, Znew).
%
%   COVARIANCETYPE selects the case of PR_ch2 Sec. 2.6:
%     'full'   (default) Case 3, Sigma_i arbitrary  -> quadratic boundary
%     'shared' Case 2, Sigma_i = Sigma (pooled)     -> linear boundary
%   REGULARIZATION (default 1e-6) adds REGULARIZATION * mean(diag(Sigma))
%   to the diagonal of each covariance. Sigma_i is singular when the
%   class has no more samples than features (PR_ch3 p. 84), which happens
%   for PCA features with d' close to n/2.
%
%   Formulas:
%     mu_i    = (1/n_i) * sum x_k                       PR_ch3 Eq. (18)
%     Sigma_i = (1/n_i) * sum (x_k - mu_i)(x_k - mu_i)' PR_ch3 Eq. (19)
%     P(w_i)  = n_i / n
%   The discriminant functions are evaluated in PREDICT_BAYES.

y = y(:);
if size(Z, 1) ~= numel(y)
    error('The number of rows in Z must match the number of labels in y.');
end
if nargin < 3 || isempty(covarianceType)
    covarianceType = 'full';
end
if nargin < 4 || isempty(regularization)
    regularization = 1e-6;
end
covarianceType = validatestring(covarianceType, {'full', 'shared'});

classes = [0; 1];
classNames = {'man'; 'woman'};
d = size(Z, 2);
n = numel(y);
mu = zeros(2, d);
Sigma = zeros(d, d, 2);
prior = zeros(2, 1);
for i = 1:2
    Zi = Z(y == classes(i), :);
    ni = size(Zi, 1);
    if ni < 2
        error('At least two %s samples are required.', classNames{i});
    end
    mu(i, :) = mean(Zi, 1);
    Zc = Zi - mu(i, :);
    Sigma(:, :, i) = (Zc.' * Zc) / ni;
    prior(i) = ni / n;
end

if strcmp(covarianceType, 'shared')
    pooled = Sigma(:, :, 1) * prior(1) + Sigma(:, :, 2) * prior(2);
    Sigma = repmat(pooled, 1, 1, 2);
end

for i = 1:2
    S = Sigma(:, :, i);
    S = (S + S.') / 2;
    Sigma(:, :, i) = S + regularization * mean(diag(S)) * eye(d);
end

model = struct( ...
    'method', 'Gaussian Bayes', ...
    'covarianceType', covarianceType, ...
    'regularization', regularization, ...
    'classes', classes, ...
    'classNames', {classNames}, ...
    'mu', mu, ...
    'Sigma', Sigma, ...
    'prior', prior, ...
    'numFeatures', d, ...
    'numParameters', countParameters(d, covarianceType));
end

function numParameters = countParameters(d, covarianceType)
% Free parameters: two means, the symmetric covariance(s), and one prior.
numCovariance = d * (d + 1) / 2;
if strcmp(covarianceType, 'full')
    numCovariance = 2 * numCovariance;
end
numParameters = 2 * d + numCovariance + 1;
end
