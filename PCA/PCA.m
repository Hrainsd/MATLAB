%% 主成分分析
clc;
clear;

A = input('请输入矩阵（m个样本n个指标的m×n矩阵）：');
[m, n] = size(A);

% 中心化数据
A_centered = A - mean(A);

% 计算协方差矩阵
cov_matrix = cov(A_centered);

% 对协方差矩阵进行奇异值分解
[U, S, V] = svd(cov_matrix);

B = input('请输入累计贡献率要求值（例如：0.8）：');

% 计算特征值
lmd = (diag(S).^2) / (m - 1);
disp('特征值为：');
disp(lmd);

% 计算主成分贡献率
ctn = lmd / sum(lmd);

% 计算主成分累计贡献率
cum_ctn = cumsum(ctn);

disp('贡献率为：');
disp(ctn);
disp('累计贡献率为：');
disp(cum_ctn);

num = find(cum_ctn >= B, 1); % 找到满足累计贡献率要求的主成分个数

disp(['只需要取前', num2str(num), '个主成分进行分析']);

% 获取主成分特征向量矩阵
disp('主成分特征向量为：');
disp(V);
PCs = V(:, 1:num);
disp('所需主成分特征向量矩阵为：');
disp(PCs);

% 以公式的形式显示主成分特征向量矩阵
fprintf('所需主成分特征向量矩阵公式：\n');
for i = 1:num
    fprintf('PC%d = ', i);
    fprintf('[');
    fprintf('%.4f ', PCs(:, i)');
    fprintf(']\n');
end
