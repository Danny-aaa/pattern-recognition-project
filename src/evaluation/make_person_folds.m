% 【檔案功能】評估：依「人」把樣本分配到交叉驗證的各折。
%   每個人有兩張照片，同一個人的兩張一定分在同一折，避免測試折出現訓練時
%   看過的臉；男、女分開分配，讓每一折的男女比例相同。

function foldId = make_person_folds(fileNames, y, numFolds, seed)
%MAKE_PERSON_FOLDS Assign samples to cross-validation folds by person.
%   FOLDID = MAKE_PERSON_FOLDS(FILENAMES, Y, NUMFOLDS, SEED) returns a fold
%   number (1..NUMFOLDS) for every sample. Files are named
%   <man|woman>_<person>_<shot>.png and each person has two shots; both
%   shots of a person go to the same fold, so the test fold never contains
%   a face seen during training. People are dealt to folds separately for
%   each gender, so every fold has the same man/woman balance.

if nargin < 3 || isempty(numFolds)
    numFolds = 5;
end
if nargin < 4 || isempty(seed)
    seed = 1;
end

y = y(:);
personId = regexprep(fileNames(:), '_\d+\.png$', '');
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
