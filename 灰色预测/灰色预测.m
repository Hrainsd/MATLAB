% 灰色预测GM(1,1)数据期数要大于3且小于10

%%
clc; clear;

% 输入数据
X = (2010:2020)'; % 年份
A = [3693, 3784, 3841, 3885, 3945, 3984, 4016, 4065, 4104, 4137, 4161]'; % 人口数据

% 数据处理
[n, m] = size(A);
A_CUM = cumsum(A); % 累计数据
Z = (A_CUM(1:n-1) + A_CUM(2:n)) / 2; % 平均数据
B = [-Z, ones(n-1, 1)]; % 设计矩阵
Y = A(2:n); % 输出数据

% 估计参数
P = (B' * B) \ (B' * Y); % 最小二乘法
a = P(1); b = P(2);

% 输入预测期数
num = input('请输入向后预测的期数（3-10）：');
if num < 3 || num > 10
    error('预测期数必须在3到10之间');
end

% 预测数据
pre_1 = zeros(n + num, 1);
for i = 1:n + num
    pre_1(i) = (A(1) - b/a) * exp(-a * (i-1)) + b/a;
end

pre_0 = zeros(n + num, 1);
pre_0(1) = A(1);
for i = 1:n + num - 1
    pre_0(i + 1) = pre_1(i + 1) - pre_1(i);
end

% 级比检验
lmd = A(1:end-1) ./ A(2:end);
JDE = exp(-2/(n+1)) < lmd & lmd < exp(2/(n+2));
if all(JDE)
    fprintf('数据通过级比检验\n');
else
    fprintf('数据未通过级比检验，选择其他方法进行预测\n');
end

% 残差检验
xiangduicancha = (A - pre_0(1:n)) ./ A; % 相对残差
count_better = sum(abs(xiangduicancha) < 0.1);
count = sum(abs(xiangduicancha) < 0.2);

if count == n && count_better == n
    fprintf('数据通过残差检验, 且达到较高要求\n');
elseif count == n
    fprintf('数据通过残差检验, 但未达到较高要求\n');
else
    fprintf('数据未通过残差检验，谨慎使用\n');
end

% 级比偏差值检验
jibipiancha = 1 - ((1 - 0.5 * a) / (1 + 0.5 * a)) .* lmd;
count_better = sum(jibipiancha < 0.1);
count = sum(jibipiancha < 0.2);

if count == n-1 && count_better == n-1
    fprintf('数据通过级比偏差值检验, 且达到较高要求\n');
elseif count == n-1
    fprintf('数据通过级比偏差值检验, 但未达到较高要求\n');
else
    fprintf('数据未通过级比偏差值检验，谨慎使用\n');
end

% 预测相对误差
err = abs((A - pre_0(1:n)) ./ A);
ave_err = mean(err);
fprintf('平均相对误差: %.2f%%\n', ave_err * 100);

% 预测数据
X1 = (2021:2021 + num - 1)'; % 预测年份
X_combined = [X; X1]; % 合并原始年份和预测年份
pre_combined = [pre_0(1:n); pre_0(n+1:end)]; % 合并原始数据和预测数据

% 绘图
figure;
plot(X, A, '-r*', 'DisplayName', '原始数据'); % 原始数据
hold on;
plot(X_combined, pre_combined, '--bo', 'DisplayName', '预测曲线'); % 所有预测数据

% 多项式拟合
degree = 5;
coeff = polyfit(X, A, degree);
X_fit = min(X):0.01:max(X);
Y_fit = polyval(coeff, X_fit);
plot(X_fit, Y_fit, '-g', 'DisplayName', '拟合曲线');

legend('Location', 'northwest');
title('灰色预测模型');
xlabel('年份');
ylabel('人口数（单位：千万人）');
grid on;

% 级比偏差值变化曲线
figure;
plot(X(2:end), jibipiancha, 'b');
xlabel('年份');
ylabel('级比偏差值');
title('级比偏差变化曲线');
grid on;

% 相对残差值变化曲线
figure;
plot(X(2:end), xiangduicancha(2:end), 'b');
xlabel('年份');
ylabel('相对残差值');
title('相对残差变化曲线');
grid on;
