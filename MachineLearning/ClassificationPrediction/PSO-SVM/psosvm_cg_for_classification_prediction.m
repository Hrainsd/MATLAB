function [best_CVacy, best_c, best_g, p_opt] = psosvm_cg_for_classification_prediction(t_train, p_train, p_opt)

%%  参数初始化
if nargin == 2
    p_opt = struct('c1', 1.5, 'c2', 1.7, 'maxgen', 10, 'sizepop', 10, ...
        'k', 0.6, 'wV', 1, 'wP', 1, 'v', 5, ...
        'popcmax', 10, 'popcmin', 10^(-1), 'popgmax', 10, 'popgmin', 10^(-1));
end

%%  设置最大速度
vcmax = p_opt.k * p_opt.popcmax;
vcmin = -vcmax;
vgmax = p_opt.k * p_opt.popgmax;
vgmin = -vgmax;

%%  误差阈值
eps = 10^(-10);

%%  种群初始化
for i = 1 : p_opt.sizepop
    
    % 随机产生种群和速度
    pp(i, 1) = (p_opt.popcmax - p_opt.popcmin) * rand + p_opt.popcmin;
    pp(i, 2) = (p_opt.popgmax - p_opt.popgmin) * rand + p_opt.popgmin;
    v(i, 1) = vcmax * rands(1, 1);
    v(i, 2) = vgmax * rands(1, 1);
    
    % 计算初始适应度
    mm = [' -v ', num2str(p_opt.v), ' -c ',num2str(pp(i, 1)), ' -g ', num2str(pp(i, 2))];
    fts(i) = (100 - svmtrain(t_train, p_train, mm)) / 100;
end

%%  初始化极值和极值点
[gb_fts, best_ind] = min(fts);   % 全局极值
lc_fts = fts;                      % 个体极值初始化
gb_x = pp(best_ind, :);                 % 全局极值点
lc_x = pp;                                % 个体极值点初始化

%%  平均适应度
avgfts_g = zeros(1, p_opt.maxgen);

%%  迭代寻优
for i = 1 : p_opt.maxgen
    for j = 1 : p_opt.sizepop
        
       % 速度更新
        v(j, :) = p_opt.wV * v(j, :) + p_opt.c1 * rand * (lc_x(j, :) ...
            - pp(j, :)) + p_opt.c2 * rand * (gb_x - pp(j, :));
        
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
        pp(j, :) = pp(j, :) + p_opt.wP * v(j, :);

        if pp(j, 1) > p_opt.popcmax
           pp(j, 1) = p_opt.popcmax;
        end

        if pp(j, 1) < p_opt.popcmin
           pp(j, 1) = p_opt.popcmin;
        end

        if pp(j, 2) > p_opt.popgmax
           pp(j, 2) = p_opt.popgmax;
        end

        if pp(j, 2) < p_opt.popgmin
           pp(j, 2) = p_opt.popgmin;
        end
        
       % 自适应粒子变异
        if rand > 0.5
            k = ceil(2 * rand);

            if k == 1
                pp(j, k) = (20 - 1) * rand + 1;
            end
            
            if k == 2
                pp(j, k) = (p_opt.popgmax - p_opt.popgmin) * rand + p_opt.popgmin;
            end

        end
        
       % 适应度值
       mm = [' -v ', num2str(p_opt.v), ' -c ', num2str(pp(j, 1)), ' -g ', num2str(pp(j, 2))];
       fts(j) = (100 - svmtrain(t_train, p_train, mm)) / 100;
        
       % 个体最优更新
        if fts(j) < lc_fts(j)
            lc_x(j, :) = pp(j, :);
            lc_fts(j) = fts(j);
        end
        
        if abs(fts(j)-lc_fts(j)) <= eps && pp(j, 1) < lc_x(j, 1)
            lc_x(j, :) = pp(j, :);
            lc_fts(j) = fts(j);
        end
        
       % 群体最优更新
        if fts(j) < gb_fts
            gb_x = pp(j, :);
            gb_fts = fts(j);
        end
        
        if abs(fts(j) - gb_fts) <= eps && pp(j, 1) < gb_x(1)
            gb_x = pp(j, :);
            gb_fts = fts(j);
        end
        
    end
    
    % 平均适应度和最佳适应度
    f_g(i) = gb_fts;
    avgfts_g(i) = sum(fts) / p_opt.sizepop;

end

%%  适应度曲线
figure
plot(1 : length(f_g), f_g, 'b-', 'LineWidth', 1.5);
title ('适应度曲线', 'FontSize', 13);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度', 'FontSize', 10);
grid on;

%%  最优值赋值
best_c = gb_x(1);
best_g = gb_x(2);
best_CVacy = (1 - f_g(p_opt.maxgen)) * 100;