%%灰狼优化算法

%%
clc;clear

% 设置灰狼优化算法的参数
obj_func = @(x) (x + 10 * sin(5 * x) + 7 * cos(4 * x));  % 求目标函数最小/大值
dim = 1;                             % 问题维度
lb = -20;                              % 搜索空间下限
ub = 20;                             % 搜索空间上限
max_iter = 1200;                     % 最大迭代次数
pack_size = 600;                     % 狼群规模

% 调用灰狼优化算法函数
[best_solutionmin, best_fitnessmin] = Grey_Wolf_Min_Fun(obj_func, dim, lb, ub, max_iter, pack_size);
[best_solutionmax, best_fitnessmax] = Grey_Wolf_Max_Fun(obj_func, dim, lb, ub, max_iter, pack_size);

% 输出最优解和最佳适应值（保留10位小数）
disp(['最优解（最小值）：', sprintf('%.10f', best_solutionmin)]);
disp(['最佳适应值（最小值）：', sprintf('%.10f', best_fitnessmin)]);
disp(['最优解（最大值）：', sprintf('%.10f', best_solutionmax)]);
disp(['最佳适应值（最大值）：', sprintf('%.10f', best_fitnessmax)]);

% % 输出最优解和最佳适应值
% disp(['最优解（最小值）：', num2str(best_solutionmin)]);
% disp(['最佳适应值（最小值）：', num2str(best_fitnessmin)]);
% disp(['最优解（最大值）：', num2str(best_solutionmax)]);
% disp(['最佳适应值（最大值）：', num2str(best_fitnessmax)]);

% 画出目标函数的图像
x = linspace(lb, ub, 1000); % 生成一系列x值
y = obj_func(x); % 计算对应的目标函数值

figure;
plot(x, y, 'LineWidth', 2, 'Color', '#FFB6C1'); 
hold on;

% 在图像中显示最小值和最大值的点
plot(best_solutionmin, best_fitnessmin, 'Color', '#70C1B3', 'Marker', '*', 'MarkerSize', 10); 
plot(best_solutionmax, best_fitnessmax, 'Color', '#9C8FBC', 'Marker', '*', 'MarkerSize', 10); 

% 设置标签
xlabel('x');
ylabel('目标函数值', 'FontName', 'SimHei', 'FontSize', 14); 
title('目标函数及最优值', 'FontName', 'SimHei', 'FontSize', 16); 

% 设置图例
legend('目标函数', '最小值', '最大值', 'Location', 'NorthWest', 'FontName', 'SimHei', 'FontSize', 6); % 使用宋体字体，调整位置和字体大小
saveas(gcf, '目标函数及最优值.svg');

hold off;

%%
% 改进的Tent混沌初始化、自适应权重计算、控制参数a和头狼位置更新方式的灰狼优化算法

% 设置灰狼优化算法的参数
obj_func = @(x) (x + 10 * sin(5 * x) + 7 * cos(4 * x));  % 求目标函数最小/大值
dim = 1;                             % 问题维度
lb = -20;                              % 搜索空间下限
ub = 20;                             % 搜索空间上限
max_iter = 1200;                     % 最大迭代次数
pack_size = 600;                     % 狼群规模

% 调用改进的灰狼优化算法函数
[best_solution_min, best_fitness_min, best_solution_max, best_fitness_max] = Improved_Grey_Wolf_Optimization(obj_func, dim, lb, ub, max_iter, pack_size);

% 输出最优解和最佳适应值（保留10位小数）
disp(['最优解（最小值）：', sprintf('%.10f', best_solution_min)]);
disp(['最佳适应值（最小值）：', sprintf('%.10f', best_fitness_min)]);
disp(['最优解（最大值）：', sprintf('%.10f', best_solution_max)]);
disp(['最佳适应值（最大值）：', sprintf('%.10f', best_fitness_max)]);

% % 输出最优解和最佳适应值
% disp(['最优解（最小值）：', num2str(best_solution_min)]);
% disp(['最佳适应值（最小值）：', num2str(best_fitness_min)]);
% disp(['最优解（最大值）：', num2str(best_solution_max)]);
% disp(['最佳适应值（最大值）：', num2str(best_fitness_max)]);

% 画出目标函数的图像
x = linspace(lb, ub, 1000); % 生成一系列x值
y = obj_func(x); % 计算对应的目标函数值

figure;
plot(x, y, 'LineWidth', 2, 'Color', '#FFB6C1'); 
hold on;

% 在图像中显示最小值和最大值的点
plot(best_solution_min, best_fitness_min, 'Color', '#70C1B3', 'Marker', '*', 'MarkerSize', 10); 
plot(best_solution_max, best_fitness_max, 'Color', '#9C8FBC', 'Marker', '*', 'MarkerSize', 10); 

% 设置标签
xlabel('x');
ylabel('目标函数值', 'FontName', 'SimHei', 'FontSize', 14); 
title('目标函数及最优值', 'FontName', 'SimHei', 'FontSize', 16); 

% 设置图例
legend('目标函数', '最小值', '最大值', 'Location', 'NorthWest', 'FontName', 'SimHei', 'FontSize', 6); % 使用宋体字体，调整位置和字体大小
saveas(gcf, '目标函数及最优值_改进.svg');

hold off;
