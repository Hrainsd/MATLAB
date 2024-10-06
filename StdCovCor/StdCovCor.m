% 创建一个示例矩阵
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12];

% 标准化矩阵
A_normalized = zscore(A);

% 计算协方差矩阵
cov_matrix = cov(A);

% 计算相关系数矩阵
corr_matrix = corrcoef(A);

% 显示结果
disp('标准化后的矩阵：');
disp(A_normalized);

disp('协方差矩阵：');
disp(cov_matrix);

disp('相关系数矩阵：');
disp(corr_matrix);
