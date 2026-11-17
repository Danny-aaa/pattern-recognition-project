% 【檔案功能】回傳專題各資料夾的絕對路徑（專題根目錄、data、data/train、results）。
%   其他程式都透過這個函式取得路徑，因此不論 MATLAB 目前在哪個資料夾都能執行。

function paths = project_paths()
%PROJECT_PATHS Default folders of the PR project.
%   PATHS = PROJECT_PATHS() returns the project root and its data, train,
%   and results folders, independent of the current MATLAB folder.

root = fileparts(fileparts(mfilename('fullpath')));
paths = struct( ...
    'root', root, ...
    'data', fullfile(root, 'data'), ...
    'train', fullfile(root, 'data', 'train'), ...
    'results', fullfile(root, 'results'));

% The result files are written here; create the folder on a fresh checkout.
if ~isfolder(paths.results)
    mkdir(paths.results);
end
end
