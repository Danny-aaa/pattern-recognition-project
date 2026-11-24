% 【檔案功能】分類器：訓練高斯貝氏分類器（第六週）。
%   用降維後的特徵估計男、女兩類各自的平均向量、共變異數矩陣與事前機率。
%   共變異數可選各類獨立（'full'，二次決策邊界）或兩類共用（'shared'，
%   線性決策邊界）。預設用 Ledoit-Wolf 收縮估計共變異數：維度接近或超過
%   每類樣本數時，樣本共變異數會奇異，收縮量會依 d / n 自動調整。

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
%   REGULARIZATION selects how each covariance is made invertible.
%   Sigma_i is singular when the class has no more samples than features
%   (PR_ch3 p. 84), which happens for PCA features with d' close to n/2.
%     'ledoit-wolf' (default) shrink the MLE toward mu * I with the
%                   data-driven Ledoit-Wolf intensity (see
%                   LEDOIT_WOLF_COVARIANCE). The shrinkage is ~0 when
%                   n_i >> d' and grows as d' approaches n_i.
%     scalar r      add r * mean(diag(Sigma)) to the diagonal (the old
%                   fixed ridge, e.g. 1e-6). With d' >= n_i this leaves
%                   d' - n_i + 1 directions with almost zero variance, and
%                   test samples are then dominated by those directions.
%   MODEL.shrinkage holds the Ledoit-Wolf intensity of each covariance
%   (NaN for a scalar REGULARIZATION).
%
%   Formulas:
%     mu_i    = (1/n_i) * sum x_k                       PR_ch3 Eq. (18)
%     Sigma_i = (1/n_i) * sum (x_k - mu_i)(x_k - mu_i)' PR_ch3 Eq. (19)
%               (then shrunk or ridged as above)
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
    regularization = 'ledoit-wolf';
end
covarianceType = validatestring(covarianceType, {'full', 'shared'});
useShrinkage = ischar(regularization) || isstring(regularization);
if useShrinkage
    regularization = validatestring(regularization, {'ledoit-wolf'});
else
    validateattributes(regularization, {'numeric'}, ...
        {'scalar', 'nonnegative', 'finite'});
end

classes = [0; 1];
classNames = {'man'; 'woman'};
d = size(Z, 2);
n = numel(y);
mu = zeros(2, d);
prior = zeros(2, 1);
centered = cell(2, 1);
for i = 1:2
    Zi = Z(y == classes(i), :);
    ni = size(Zi, 1);
    if ni < 2
        error('At least two %s samples are required.', classNames{i});
    end
    mu(i, :) = mean(Zi, 1);
    centered{i} = Zi - mu(i, :);
    prior(i) = ni / n;
end

% 'full' estimates one covariance per class; 'shared' pools the centered
% samples of both classes, which gives Sigma = P(w_1) Sigma_1 + P(w_2) Sigma_2.
if strcmp(covarianceType, 'full')
    groups = {centered{1}, centered{2}};
else
    groups = {[centered{1}; centered{2}]};
end
Sigma = zeros(d, d, 2);
shrinkage = nan(2, 1);
for g = 1:numel(groups)
    if useShrinkage
        [S, alpha] = ledoit_wolf_covariance(groups{g});
    else
        Zc = groups{g};
        S = (Zc.' * Zc) / size(Zc, 1);
        S = (S + S.') / 2;
        S = S + regularization * mean(diag(S)) * eye(d);
        alpha = NaN;
    end
    if isscalar(groups)
        Sigma = repmat(S, 1, 1, 2);
        shrinkage(:) = alpha;
    else
        Sigma(:, :, g) = S;
        shrinkage(g) = alpha;
    end
end

model = struct( ...
    'method', 'Gaussian Bayes', ...
    'covarianceType', covarianceType, ...
    'regularization', regularization, ...
    'shrinkage', shrinkage, ...
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
