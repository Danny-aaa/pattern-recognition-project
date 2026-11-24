% 【檔案功能】評估：把實驗選項中的共變異數種類整理成標準寫法。
%   fit_bayes 用 validatestring 接受大小寫不同或縮寫（例如 'Full'、'sh'），
%   這裡先換成標準名稱 'full' / 'shared'，結果表格與圖例才不會標錯。

function types = validate_covariance_types(types)
%VALIDATE_COVARIANCE_TYPES Canonical names of Bayes covariance types.
%   TYPES = VALIDATE_COVARIANCE_TYPES(TYPES) accepts a char vector, a
%   string array, or a cell array of them and returns a cell row vector
%   with each entry replaced by 'full' or 'shared', as FIT_BAYES reads it.

if ischar(types) || isstring(types)
    types = cellstr(types);
end
types = cellfun(@(t) validatestring(t, {'full', 'shared'}), types(:).', ...
    'UniformOutput', false);
end
