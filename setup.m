% 【檔案功能】專題環境設定腳本：把 src/ 底下所有程式資料夾加入 MATLAB 路徑。
%   每次開啟 MATLAB 後先執行一次，之後才能在任何資料夾呼叫專題的函式。
%
%   專題資料夾說明：
%     src/preprocessing  前處理：影像轉特徵向量、計算各類別的平均與共變異數
%     src/features       特徵降維：PCA、LDA、scatter matrix 分析（第九週）
%     src/classifiers    分類器：高斯貝氏分類器的訓練與預測
%     src/evaluation     評估：依人分組的五折交叉驗證、ROC 與 AUC、
%                        PCA 維度實驗（第十週）、LDA 維度實驗（第十一週）
%     data/train         200 張訓練人臉影像（男 100、女 100）
%     data/*.mat         由前處理程式產生的訓練向量與類別共變異數
%                        （不在 GitHub 上，第一次使用請先執行 build_training_vectors）
%     results/           實驗輸出：交叉驗證結果、ROC 圖、eigenfaces、特徵值圖
%   完整的使用說明與目前進度請看 README.md。
%
%SETUP Add the PR project code to the MATLAB path.
%   Run this once per MATLAB session, from the project folder:
%     setup
%   or from any folder:
%     run('<path to the project>/setup.m')
%
%   Folders:
%     src/preprocessing  images -> feature vectors, class covariances
%     src/features       PCA and LDA
%     src/classifiers    Bayes classifier
%     src/evaluation     folds, ROC/AUC, cross-validation
%     data/              training images and generated .mat data
%     results/           cross-validation results and figures

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'src')));
