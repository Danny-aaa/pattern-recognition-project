% 【檔案功能】評估：計算 ROC 曲線與曲線下面積 AUC（以女性為正類）。
%   把判斷門檻掃過所有分數，得到命中率（true-positive rate）與
%   誤警率（false-positive rate），再用梯形法求 AUC。

function [falseAlarm, hit, auc, thresholds] = roc_auc(score, y)
%ROC_AUC ROC curve and area under it, with woman (y = 1) as the positive class.
%   [FALSEALARM, HIT, AUC, THRESHOLDS] = ROC_AUC(SCORE, Y) sweeps the
%   decision threshold x* over every distinct SCORE and returns
%     hit         = P(score > x* | woman)   (true-positive rate)
%     false alarm = P(score > x* | man)     (false-positive rate)
%   as in PR_ch2 Sec. 2.8.3, pp. 135-137. The curve starts at (0, 0) and
%   ends at (1, 1). AUC is the trapezoidal area under the curve; tied
%   scores are handled by moving through them in one step.

score = score(:);
y = y(:);
numPositive = nnz(y == 1);
numNegative = nnz(y == 0);
if numPositive == 0 || numNegative == 0
    error('Both classes are required to compute an ROC curve.');
end

[sortedScore, order] = sort(score, 'descend');
ySorted = y(order);

% Keep the last index of each run of equal scores.
lastOfTie = [diff(sortedScore) ~= 0; true];
truePositives = cumsum(ySorted == 1);
falsePositives = cumsum(ySorted == 0);

hit = [0; truePositives(lastOfTie) / numPositive];
falseAlarm = [0; falsePositives(lastOfTie) / numNegative];
thresholds = [inf; sortedScore(lastOfTie)];
auc = trapz(falseAlarm, hit);
end
