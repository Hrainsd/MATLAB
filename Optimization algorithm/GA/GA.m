%%遗传算法

%%
clc;clear

% 目标函数
obj_func = @(x) (x + 10 * sin(5 * x) + 7 * cos(4 * x));

% 遗传算法参数
pop_size = 800;  % 种群大小
num_generations = 1200;  % 迭代次数
crossover_prob = 0.8;  % 交叉概率
mutation_prob = 0.4;  % 变异概率
num_selections = pop_size;  % 选择个体数量
optimizationTypeMin = 'min';  % 计算最小值
optimizationTypeMax = 'max';  % 计算最大值

% 初始化种群
lb = -20;  % 适应度函数定义域下限
ub = 20;   % 适应度函数定义域上限
population = (ub - lb) * rand(pop_size, 1) + lb;  % 在[lb, ub]范围内随机初始化种群

% 存储每一代的最小和最大适应值
min_fitness_history_min = zeros(num_generations, 1);
max_fitness_history_min = zeros(num_generations, 1);

min_fitness_history_max = zeros(num_generations, 1);
max_fitness_history_max = zeros(num_generations, 1);

% 遗传算法主循环 - 计算最小值
for generation = 1:num_generations
    % 评估种群中每个个体的适应度
    fitness = obj_func(population);
    
    % 记录最小和最大适应值
    min_fitness_history_min(generation) = min(fitness);
    max_fitness_history_min(generation) = max(fitness);
    
    % 使用整合的遗传算法函数
    population_min = genetic_algorithm(population, fitness, crossover_prob, mutation_prob, num_selections, optimizationTypeMin);
end

% 遗传算法主循环 - 计算最大值
population_max = (ub - lb) * rand(pop_size, 1) + lb;  % 重新初始化种群
for generation = 1:num_generations
    % 评估种群中每个个体的适应度
    fitness = obj_func(population_max);
    
    % 记录最小和最大适应值
    min_fitness_history_max(generation) = min(fitness);
    max_fitness_history_max(generation) = max(fitness);
    
    % 使用整合的遗传算法函数
    population_max = genetic_algorithm(population_max, fitness, crossover_prob, mutation_prob, num_selections, optimizationTypeMax);
end

% 寻找最优解和最佳适应值（最小值）
best_solution_min = population_min(1);
best_fitness_min = obj_func(best_solution_min);

% 寻找最优解和最佳适应值（最大值）
best_solution_max = population_max(1);
best_fitness_max = obj_func(best_solution_max);

% 输出最优解和最佳适应值（保留10位小数）
disp(['最小值的最优解：', sprintf('%.10f', best_solution_min)]);
disp(['最小值的最佳适应值：', sprintf('%.10f', best_fitness_min)]);
disp(['最大值的最优解：', sprintf('%.10f', best_solution_max)]);
disp(['最大值的最佳适应值：', sprintf('%.10f', best_fitness_max)]);

% 绘制适应度进化曲线
figure;
plot(1:num_generations, min_fitness_history_min, 'Color', '#D291BC', 'LineWidth', 2);
hold on;
plot(1:num_generations, max_fitness_history_min, 'Color', '#FF6F61', 'LineWidth', 2);
xlabel('代数');
ylabel('适应度');
title('适应度进化曲线 - 最小值');
legend('最小适应度', '最大适应度');
grid on;
hold off;
saveas(gcf, '适应度进化曲线_最小值.svg');

figure;
plot(1:num_generations, min_fitness_history_max, 'Color', '#D291BC', 'LineWidth', 2);
hold on;
plot(1:num_generations, max_fitness_history_max, 'Color', '#FF6F61', 'LineWidth', 2);
xlabel('代数');
ylabel('适应度');
title('适应度进化曲线 - 最大值');
legend('最小适应度', '最大适应度');
grid on;
hold off;
saveas(gcf, '适应度进化曲线_最大值.svg');

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
legend('目标函数', '最小值', '最大值', 'Location', 'NorthWest', 'FontName', 'SimHei', 'FontSize', 6);
saveas(gcf, '目标函数及最优值.svg');
hold off;

%%
% 自适应遗传算法
% 目标函数
obj_func = @(x) (x + 10 * sin(5 * x) + 7 * cos(4 * x));

% 自适应遗传算法参数
pop_size = 800;  % 初始种群大小
num_generations = 1200;  % 迭代次数
crossover_prob = 0.8;  % 交叉概率
mutation_prob_min = 0.2;  % 最小变异概率
mutation_prob_max = 0.9;  % 最大变异概率
num_selections = pop_size;  % 选择个体数量
optimizationTypeMin = 'min';  % 计算最小值
optimizationTypeMax = 'max';  % 计算最大值

% 初始化种群
lb = -20;  % 适应度函数定义域下限
ub = 20;   % 适应度函数定义域上限
population = (ub - lb) * rand(pop_size, 1) + lb;  % 在[lb, ub]范围内随机初始化种群

% 存储每一代的最小和最大适应值
min_fitness_history_min = zeros(num_generations, 1);
max_fitness_history_min = zeros(num_generations, 1);

min_fitness_history_max = zeros(num_generations, 1);
max_fitness_history_max = zeros(num_generations, 1);

% 自适应遗传算法主循环
for generation = 1:num_generations
    % 评估种群中每个个体的适应度
    fitness = obj_func(population);
    
    % 记录最小和最大适应值
    min_fitness_history_min(generation) = min(fitness);
    max_fitness_history_min(generation) = max(fitness);
    
    % 使用改进的自适应遗传算法函数 - 计算最小值
    population_min = adaptive_genetic_algorithm(population, fitness, crossover_prob, mutation_prob_min, num_selections, optimizationTypeMin);
    
    % 记录最小和最大适应值
    min_fitness_history_min(generation) = min(fitness);
    max_fitness_history_min(generation) = max(fitness);
end

for generation = 1:num_generations
    % 评估种群中每个个体的适应度
    fitness = obj_func(population);
    % 记录最小和最大适应值
    min_fitness_history_max(generation) = min(fitness);
    max_fitness_history_max(generation) = max(fitness);

    % 使用改进的自适应遗传算法函数 - 计算最大值
    population_max = adaptive_genetic_algorithm(population, fitness, crossover_prob, mutation_prob_max, num_selections, optimizationTypeMax);
    
    % 记录最小和最大适应值
    min_fitness_history_max(generation) = min(fitness);
    max_fitness_history_max(generation) = max(fitness);
end

% 合并最小值和最大值的种群
population = [population_min; population_max];

% 寻找最优解和最佳适应值（最小值）
best_solution_min = population(1);
best_fitness_min = obj_func(best_solution_min);

% 寻找最优解和最佳适应值（最大值）
best_solution_max = population(end);
best_fitness_max = obj_func(best_solution_max);

% 输出最优解和最佳适应值（保留10位小数）
disp(['最小值的最优解：', sprintf('%.10f', best_solution_min)]);
disp(['最小值的最佳适应值：', sprintf('%.10f', best_fitness_min)]);

disp(['最大值的最优解：', sprintf('%.10f', best_solution_max)]);
disp(['最大值的最佳适应值：', sprintf('%.10f', best_fitness_max)]);

% 画出目标函数的图像
x = linspace(lb, ub, 1000); % 生成一系列x值
y = obj_func(x); % 计算对应的目标函数值

% 绘制适应度进化曲线 - 最小值
figure;
plot(1:num_generations, min_fitness_history_min, 'Color', '#D291BC', 'LineWidth', 2);
hold on;
plot(1:num_generations, max_fitness_history_min, 'Color', '#FF6F61', 'LineWidth', 2);
xlabel('代数');
ylabel('适应度');
title('适应度进化曲线 - 最小值');
legend('最小适应度', '最大适应度');
grid on;
hold off;
saveas(gcf, '适应度进化曲线_最小值_改进.svg');

% 绘制适应度进化曲线 - 最大值
figure;
plot(1:num_generations, min_fitness_history_max, 'Color', '#D291BC', 'LineWidth', 2);
hold on;
plot(1:num_generations, max_fitness_history_max, 'Color', '#FF6F61', 'LineWidth', 2);
xlabel('代数');
ylabel('适应度');
title('适应度进化曲线 - 最大值');
legend('最小适应度', '最大适应度');
grid on;
hold off;
saveas(gcf, '适应度进化曲线_最大值_改进.svg');

% 画出目标函数的图像
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
