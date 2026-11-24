% 【檔案功能】評估：依「人」把樣本分配到交叉驗證的各折。
%   每個人有兩張照片，同一個人的兩張一定分在同一折，避免測試折出現訓練時
%   看過的臉；男、女分開分配，讓每一折的男女比例相同。
%   檔名必須符合 <man|woman>_<人>_<張>.<副檔名>，否則直接報錯：若檔名無法
%   解析，同一人的照片可能被當成不同人而分到不同折，造成資料洩漏。

function foldId = make_person_folds(fileNames, y, numFolds, seed)
%MAKE_PERSON_FOLDS Assign samples to cross-validation folds by person.
%   FOLDID = MAKE_PERSON_FOLDS(FILENAMES, Y, NUMFOLDS, SEED) returns a fold
%   number (1..NUMFOLDS) for every sample. Files are named
%   <man|woman>_<person>_<shot>.<png|jpg|jpeg|bmp> (case-insensitive) and
%   each person has two shots; both shots of a person go to the same fold,
%   so the test fold never contains a face seen during training. People
%   are dealt to folds separately for each gender, so every fold has the
%   same man/woman balance.
%
%   A file name that does not follow the pattern, or whose gender prefix
%   disagrees with Y, is an error: the person could not be identified, and
%   grouping it as a person of its own would leak that face across folds.
%   A person with a number of shots other than two gives a warning.

if nargin < 3 || isempty(numFolds)
    numFolds = 5;
end
if nargin < 4 || isempty(seed)
    seed = 1;
end

y = y(:);
fileNames = cellstr(fileNames(:));
if numel(fileNames) ~= numel(y)
    error('fileNames and y must have the same number of elements.');
end

tokens = regexpi(fileNames, ...
    '^(man|woman)_(\d+)_\d+\.(png|jpe?g|bmp)$', 'tokens', 'once');
unparsed = cellfun(@isempty, tokens);
if any(unparsed)
    error(['%d file name(s) do not match <man|woman>_<person>_<shot>.png, ' ...
        'e.g. "%s". The person of each image must be known to keep both ' ...
        'shots in the same fold.'], nnz(unparsed), ...
        fileNames{find(unparsed, 1)});
end
gender = cellfun(@(t) lower(t{1}), tokens, 'UniformOutput', false);
personId = cellfun(@(g, t) [g, '_', t{2}], gender, tokens, ...
    'UniformOutput', false);

wrongLabel = (strcmp(gender, 'woman') ~= (y == 1)) | ~ismember(y, [0, 1]);
if any(wrongLabel)
    error('The label of "%s" (y = %g) does not match its file name.', ...
        fileNames{find(wrongLabel, 1)}, y(find(wrongLabel, 1)));
end

[people, ~, personIndex] = unique(personId);
shotCount = accumarray(personIndex, 1);
if any(shotCount ~= 2)
    odd = find(shotCount ~= 2, 1);
    warning(['%d person(s) do not have exactly two images, e.g. %s has %d. ' ...
        'Their images are still kept in one fold.'], nnz(shotCount ~= 2), ...
        people{odd}, shotCount(odd));
end

foldId = zeros(numel(y), 1);
stream = RandStream('mt19937ar', 'Seed', seed);

for label = [0, 1]
    people = unique(personId(y == label));
    people = people(randperm(stream, numel(people)));
    for p = 1:numel(people)
        foldId(strcmp(personId, people{p})) = mod(p - 1, numFolds) + 1;
    end
end
end
