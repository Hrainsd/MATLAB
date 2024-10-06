%% 鲸鱼优化算法

%%
clc;clear

% 设置鲸鱼优化算法的参数
obj_func = @(x) (x.*sin(x).*cos(2*x) - 2*x.*sin(3*x) + 3*x.*sin(4*x));  % 求目标函数最小/大值
numVariables = 1; % 变量维度
numWhales = 600; % 鲸鱼数量
maxIterations = 1200; % 最大迭代次数

minValue = 0; % x的取值范围
maxValue = 50;

% 调用鲸鱼优化算法
[bestSolutionMin, bestFitnessMin] = Whale_Optimization_Algorithm_min_fun(obj_func, numVariables, numWhales, maxIterations, minValue, maxValue);
[bestSolutionMax, bestFitnessMax] = Whale_Optimization_Algorithm_max_fun(obj_func, numVariables, numWhales, maxIterations, minValue, maxValue);

% 输出最优解和最佳适应值（保留10位小数）
disp(['最优解（最小值）：', sprintf('%.10f', bestSolutionMin)]);
disp(['最佳适应值（最小值）：', sprintf('%.10f', bestFitnessMin)]);
disp(['最优解（最大值）：', sprintf('%.10f', bestSolutionMax)]);
disp(['最佳适应值（最大值）：', sprintf('%.10f', bestFitnessMax)]);

% % 输出最优解和最佳适应值
% disp(['最优解（最小值）：', num2str(bestSolutionMin)]);
% disp(['最佳适应值（最小值）：', num2str(bestFitnessMin)]);
% disp(['最优解（最大值）：', num2str(bestSolutionMax)]);
% disp(['最佳适应值（最大值）：', num2str(bestFitnessMax)]);

% 画出目标函数的图像
x = linspace(minValue, maxValue, 1000); % 生成一系列x值
y = obj_func(x); % 计算对应的目标函数值

figure;
plot(x, y,'LineWidth', 2, 'Color', '#FFB6C1'); 
hold on;

% 在图像中显示最小值和最大值的点
plot(bestSolutionMin, bestFitnessMin, 'Color', '#70C1B3', 'Marker', '*', 'MarkerSize', 10); 
plot(bestSolutionMax, bestFitnessMax, 'Color', '#9C8FBC', 'Marker', '*', 'MarkerSize', 10); 

% 设置标签
xlabel('x');
ylabel('目标函数值', 'FontName', 'SimHei', 'FontSize', 14); 
title('目标函数及最优值', 'FontName', 'SimHei', 'FontSize', 16); 

% 设置图例
legend('目标函数', '最小值', '最大值', 'Location', 'NorthWest', 'FontName', 'SimHei', 'FontSize', 6); % 使用宋体字体，调整位置和字体大小
saveas(gcf, '目标函数及最优值.svg');
hold off;

%%
% 融合模拟退火和自适应变异的混沌鲸鱼优化算法

% 设置混沌鲸鱼优化算法的参数
obj_func = @(x) (x + 10 * sin(5 * x) + 7 * cos(4 * x));  % 求目标函数最小/大值
numVariables = 1; % 变量维度
numWhales = 600; % 鲸鱼数量
maxIterations = 1200; % 最大迭代次数

minValue = -20;
maxValue = 20;

% 调用混沌鲸鱼优化算法
[bestSolutionMin, bestFitnessMin] = Chaotic_Whale_Optimization_Algorithm(obj_func, numVariables, numWhales, maxIterations, minValue, maxValue, 'min');
[bestSolutionMax, bestFitnessMax] = Chaotic_Whale_Optimization_Algorithm(obj_func, numVariables, numWhales, maxIterations, minValue, maxValue, 'max');

% 输出最优解和最佳适应值（保留10位小数）
disp(['最优解（最小值）：', sprintf('%.10f', bestSolutionMin)]);
disp(['最佳适应值（最小值）：', sprintf('%.10f', bestFitnessMin)]);
disp(['最优解（最大值）：', sprintf('%.10f', bestSolutionMax)]);
disp(['最佳适应值（最大值）：', sprintf('%.10f', bestFitnessMax)]);

% % 输出最优解和最佳适应值
% disp(['最优解（最小值）：', num2str(bestSolutionMin)]);
% disp(['最佳适应值（最小值）：', num2str(bestFitnessMin)]);
% disp(['最优解（最大值）：', num2str(bestSolutionMax)]);
% disp(['最佳适应值（最大值）：', num2str(bestFitnessMax)]);

% 画出目标函数的图像
x = linspace(minValue, maxValue, 1000); % 生成一系列x值
y = obj_func(x); % 计算对应的目标函数值

figure;
plot(x, y,'LineWidth', 2, 'Color', '#FFB6C1'); % 画出目标函数图像
hold on;

% 在图像中显示最小值和最大值的点
plot(bestSolutionMin, bestFitnessMin, 'Color', '#70C1B3', 'Marker', '*', 'MarkerSize', 10); % 最小值的点为红色
plot(bestSolutionMax, bestFitnessMax, 'Color', '#9C8FBC', 'Marker', '*', 'MarkerSize', 10); % 最大值的点为绿色

% 设置标签
xlabel('x');
ylabel('目标函数值', 'FontName', 'SimHei', 'FontSize', 14); % 使用宋体字体
title('目标函数及最优值', 'FontName', 'SimHei', 'FontSize', 16); % 使用宋体字体

% 设置图例
legend('目标函数', '最小值', '最大值', 'Location', 'NorthWest', 'FontName', 'SimHei', 'FontSize', 6); % 使用宋体字体，调整位置和字体大小
saveas(gcf, '目标函数及最优值_改进.svg');

hold off;
