% 灰狼优化算法
function [best_solution, best_fitness] = Grey_Wolf_Min_Fun(obj_func, dim, lb, ub, max_iter, pack_size)

    % 初始化种群
    alpha_pos = rand(1, dim) * (ub - lb) + lb;
    beta_pos = rand(1, dim) * (ub - lb) + lb;
    delta_pos = rand(1, dim) * (ub - lb) + lb;
    
    alpha_score = feval(obj_func, alpha_pos);
    beta_score = feval(obj_func, beta_pos);
    delta_score = feval(obj_func, delta_pos);
    
    positions = rand(pack_size, dim) * (ub - lb) + lb;
    
    % 开始迭代
    for iter = 1:max_iter
        a = 2 - iter * (2 / max_iter); % 调整参数a，用于控制狼群的收敛
        
        for i = 1:pack_size
            fitness = feval(obj_func, positions(i, :));
            
            % 更新alpha狼位置
            if fitness < alpha_score
                alpha_score = fitness;
                alpha_pos = positions(i, :);
            end
            
            % 更新beta狼位置
            if fitness > alpha_score && fitness < beta_score
                beta_score = fitness;
                beta_pos = positions(i, :);
            end
            
            % 更新delta狼位置
            if fitness > alpha_score && fitness > beta_score && fitness < delta_score
                delta_score = fitness;
                delta_pos = positions(i, :);
            end
        end
        
        for i = 1:pack_size
            r1 = rand(); % 随机数[0,1]
            r2 = rand();
            
            A1 = 2 * a * r1 - a; % 更新参数A1
            C1 = 2 * r2; % 更新参数C1
            
            D_alpha = abs(C1 * alpha_pos - positions(i, :));
            X1 = alpha_pos - A1 * D_alpha;
            
            r1 = rand();
            r2 = rand();
            
            A2 = 2 * a * r1 - a; % 更新参数A2
            C2 = 2 * r2; % 更新参数C2
            
            D_beta = abs(C2 * beta_pos - positions(i, :));
            X2 = beta_pos - A2 * D_beta;
            
            r1 = rand();
            r2 = rand();
            
            A3 = 2 * a * r1 - a; % 更新参数A3
            C3 = 2 * r2; % 更新参数C3
            
            D_delta = abs(C3 * delta_pos - positions(i, :));
            X3 = delta_pos - A3 * D_delta;
            
            % 更新狼位置
            positions(i, :) = (X1 + X2 + X3) / 3;
            
            % 边界处理
            positions(i, :) = max(positions(i, :), lb);
            positions(i, :) = min(positions(i, :), ub);
        end
        
        % 输出当前迭代的最佳适应值
%         disp(['Iteration ', num2str(iter), ': Best Fitness = ', num2str(alpha_score)]);
    end
    
    % 返回最佳解和最佳适应值
    best_solution = alpha_pos;
    best_fitness = alpha_score;
end