% 【檔案功能】評估：把使用者給的選項結構和預設值合併。
%   沒有指定的欄位使用預設值；不認得的欄位（多半是打錯字）會發出警告，
%   避免設定被默默忽略。各實驗程式（run_cross_validation、
%   pca_dimension_experiment、lda_dimension_experiment）共用這個函式。

function options = merge_options(options, defaults)
%MERGE_OPTIONS Fill the missing fields of OPTIONS from DEFAULTS.
%   OPTIONS = MERGE_OPTIONS(OPTIONS, DEFAULTS) copies every field of the
%   struct DEFAULTS that OPTIONS lacks. OPTIONS may be [] or a struct; a
%   field of OPTIONS that DEFAULTS does not have gives a warning.

if isempty(options)
    options = struct();
end
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar struct.');
end

names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(options, names{i})
        options.(names{i}) = defaults.(names{i});
    end
end

unknown = setdiff(fieldnames(options), names);
if ~isempty(unknown)
    warning('Unknown option(s) ignored: %s', strjoin(unknown, ', '));
end
end
