function [best_CVmse, best_c, best_g, p_o] = psosvm_cg_for_regression_prediction(train_label, train, p_o)

% 参数初始化
if nargin == 2
    p_o = struct('c1', 1.5, 'c2', 1.7, 'maxgen', 20, 'sizepop', 20, ...
                    'k', 0.6, 'wV', 1, 'wP', 1, 'v',5, ...
                     'popcmax', 10^2, 'popcmin', 10^(-1), 'popgmax', 10^3, 'popgmin', 10^(-2));
end

vcmax = p_o.k * p_o.popcmax; % 设置速度最大值
vcmin = -vcmax ;

vgmax = p_o.k * p_o.popgmax;
vgmin = -vgmax ;

eps = 10^(-10); % 设置误差阈值

% 产生初始粒子和速度
for i = 1 : p_o.sizepop
    
    % 随机产生种群和速度
    pp(i, 1) = (p_o.popcmax - p_o.popcmin) * rand + p_o.popcmin;  
    pp(i, 2) = (p_o.popgmax - p_o.popgmin) * rand + p_o.popgmin;
    v(i, 1)   = vcmax * rands(1, 1);  
    v(i, 2)   = vgmax * rands(1, 1);
    
    % 计算初始适应度
    mm = [' -v ', num2str(p_o.v), ' -c ', num2str(pp(i, 1)), ' -g ', ...
        num2str(pp(i, 2)), ' -s 3 -p 0.1'];

    fitness(i) = svmtrain(train_label, train, mm);

end


[gb_fitness, bestindex] = min(fitness); % 初始化极值和极值点   
lc_fitness = fitness;                      
gb_x = pp(bestindex, :);                 
lc_x = pp;                                

% 每一代种群的平均适应度
af_gen = zeros(1, p_o.maxgen); 

%  迭代寻优
for i = 1 : p_o.maxgen
    for j = 1 : p_o.sizepop
        
        % 速度更新        
        v(j, :) = p_o.wV * v(j, :) + p_o.c1 * rand * (lc_x(j, :) - pp(j, :)) + ...
            p_o.c2 * rand * (gb_x - pp(j, :));

        if v(j, 1) > vcmax
           v(j, 1) = vcmax;
        end

        if v(j, 1) < vcmin
           v(j, 1) = vcmin;
        end

        if v(j, 2) > vgmax
           v(j, 2) = vgmax;
        end

        if v(j, 2) < vgmin
           v(j, 2) = vgmin;
        end
        
        % 种群更新
        pp(j, :) = pp(j, :) + p_o.wP * v(j, :);
        
        if pp(j, 1) > p_o.popcmax
           pp(j, 1) = p_o.popcmax;
        end
        
        if pp(j, 1) < p_o.popcmin
           pp(j, 1) = p_o.popcmin;
        end
        
        if pp(j, 2) > p_o.popgmax
           pp(j, 2) = p_o.popgmax;
        end
        
        if pp(j, 2) < p_o.popgmin
           pp(j, 2) = p_o.popgmin;
        end
        
        % 自适应粒子变异
        if rand > 0.5
            k = ceil(2 * rand);

            if k == 1
               pp(j, k) = (20 - 1) * rand + 1;
            end

            if k == 2
               pp(j, k) = (p_o.popgmax - p_o.popgmin) * rand + p_o.popgmin;
            end     

        end
        
        % 适应度值
        mm = [' -v ', num2str(p_o.v), ' -c ', num2str(pp(j, 1)), ' -g ', ...
            num2str(pp(j, 2)), ' -s 3 -p 0.1'];

        fitness(j) = svmtrain(train_label, train, mm);
        
        % 个体最优更新
        if fitness(j) < lc_fitness(j)
           lc_x(j, :) = pp(j, :);
           lc_fitness(j) = fitness(j);
        end

        if fitness(j) == lc_fitness(j) && pp(j, 1) < lc_x(j, 1)
           lc_x(j, :) = pp(j, :);
           lc_fitness(j) = fitness(j);
        end        
        
        % 群体最优更新
        if fitness(j) < gb_fitness
           gb_x = pp(j, :);
           gb_fitness = fitness(j);
        end

        if abs(fitness(j) - gb_fitness) <= eps && pp(j,1) < gb_x(1)
           gb_x = pp(j, :);
           gb_fitness = fitness(j);
        end
        
    end
    
    fit_gen(i) = gb_fitness;    
    af_gen(i) = sum(fitness) / p_o.sizepop;
end

%  适应度曲线
figure
plot(1 : length(fit_gen), fit_gen, 'b-', 'LineWidth', 1.5);
title('最佳适应度曲线', 'FontSize', 13);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度', 'FontSize', 10);
xlim([1, length(fit_gen)])
grid on

%  赋值
best_c = gb_x(1);
best_g = gb_x(2);
best_CVmse = fit_gen(p_o.maxgen);
