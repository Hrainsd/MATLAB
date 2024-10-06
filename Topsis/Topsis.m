%%%
clc; clear;

% 输入样本矩阵和是否需要正向化的标志
A = input('请输入行是样本，列是指标的矩阵：'); 
[n, m] = size(A);
deter = input('您是否需要正向化（0：不需要，1：需要）：');

% 计算指标权重
weights = entropy_weight(A);

% 判断是否需要正向化
if deter == 0
    [sorted_S, index] = after_positive(A);
else
    A = minmin(A, weights);
    A = median(A, weights);
    A = interval(A, weights);
    [sorted_S, index] = after_positive(A); % sorted_s为降序排列后的得分，index为得分降序排序后的索引
end

% 输出结果
for i = 1:n
    disp(['第', num2str(index(i)), '个样本得分为：', num2str(sorted_S(i)), ',排名第', num2str(i)]);
end
