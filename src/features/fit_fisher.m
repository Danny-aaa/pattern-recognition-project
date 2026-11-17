% 【檔案功能】特徵降維：在給定的特徵空間內求兩類的 Fisher 線性鑑別方向。
%   輸入是已經降維過的特徵（例如 PCA 特徵），計算類內散佈矩陣 S_W、
%   類間散佈矩陣 S_B 與 Fisher 方向 w = S_W^-1 (c1 - c2)。
%   fit_lda 與第十一週的 LDA 維度實驗都呼叫這個函式。

function fisher = fit_fisher(A, y)
%FIT_FISHER Fisher's linear discriminant in the feature space of A.
%   FISHER = FIT_FISHER(A, Y) treats each row of A (n-by-p) as one sample
%   with label Y (0 = man, 1 = woman) and returns the unit Fisher direction
%   FISHER.w (p-by-1), so that the one-dimensional LDA feature is A * w;
%   larger values indicate the woman class. p must not exceed n - 2, or
%   S_W is singular (PR_ch3 p. 84).
%
%   Formulas (PR_ch3, Sec. 3.8.2), with class 1 = woman and class 2 = man:
%     c_i = (1/N_i) * sum_{x in class i} x                       p. 109
%     S_i = sum_{x in class i} (x - c_i)(x - c_i)'               p. 113
%     S_W = S_1 + S_2                                            p. 115
%     S_B = (c_1 - c_2)(c_1 - c_2)'                              p. 116
%     J(w) = (w' S_B w) / (w' S_W w)                             p. 117
%     w* = S_W^-1 (c_1 - c_2)                                    p. 119

y = y(:);
if size(A, 1) ~= numel(y)
    error('The number of rows in A must match the number of labels in y.');
end
isWoman = (y == 1);
isMan = (y == 0);
if nnz(isWoman) < 2 || nnz(isMan) < 2 || any(~isWoman & ~isMan)
    error('y must contain labels 0 (man) and 1 (woman), at least two of each.');
end

c1 = mean(A(isWoman, :), 1).';
c2 = mean(A(isMan, :), 1).';
A1 = A(isWoman, :) - c1.';
A2 = A(isMan, :) - c2.';
Sw = A1.' * A1 + A2.' * A2;
Sw = (Sw + Sw.') / 2;
Sb = (c1 - c2) * (c1 - c2).';

% The magnitude of w is immaterial (PR_ch3 p. 119); normalize it.
conditionEstimate = rcond(Sw);
warningState = warning('off', 'MATLAB:nearlySingularMatrix');
restoreWarning = onCleanup(@() warning(warningState));
w = Sw \ (c1 - c2);
w = w / norm(w);

fisher = struct( ...
    'w', w, ...
    'Sw', Sw, ...
    'Sb', Sb, ...
    'fisherCriterion', (w.' * Sb * w) / (w.' * Sw * w), ...
    'rcondSw', conditionEstimate);
end
