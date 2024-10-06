function new_population = genetic_algorithm(population, fitness, crossover_prob, mutation_prob, num_selections, optimizationType)
    cumulative_prob = cumsum(fitness) / sum(fitness);
    selected_indices = zeros(1, num_selections);

    for i = 1:num_selections
        rand_num = rand();
        selected_indices(i) = find(cumulative_prob >= rand_num, 1, 'first');
    end
    
    crossovered_population = crossover(population(selected_indices, :), crossover_prob);
    mutated_population = mutation(crossovered_population, mutation_prob);
    
    new_population = mutated_population;
    
    % 根据优化类型选择最大化或最小化
    if strcmpi(optimizationType, 'max')
        [~, idx] = max(fitness);
        new_population(1) = population(idx);
    elseif strcmpi(optimizationType, 'min')
        [~, idx] = min(fitness);
        new_population(1) = population(idx);
    else
        error('Invalid optimizationType. Use ''min'' or ''max''.');
    end
end

% 适应度加权轮盘赌选择
function selected_indices = roulette_wheel_selection(fitness, num_selections)
    cumulative_prob = cumsum(fitness) / sum(fitness);
    selected_indices = zeros(1, num_selections);
    
    for i = 1:num_selections
        rand_num = rand();
        selected_indices(i) = find(cumulative_prob >= rand_num, 1, 'first');
    end
end

% 交叉操作（单点交叉）
function crossovered_population = crossover(parents, crossover_prob)
    crossovered_population = zeros(size(parents));
    
    for i = 1:2:length(parents)
        if rand() < crossover_prob
            crossover_point = randi(length(parents(i, :)));
            crossovered_population(i, :) = [parents(i, 1:crossover_point), parents(i+1, crossover_point+1:end)];
            crossovered_population(i+1, :) = [parents(i+1, 1:crossover_point), parents(i, crossover_point+1:end)];
        else
            crossovered_population(i, :) = parents(i, :);
            crossovered_population(i+1, :) = parents(i+1, :);
        end
    end
end

% 变异操作（基本变异）
function mutated_population = mutation(population, mutation_prob)
    mutated_population = population;
    
    for i = 1:length(population)
        if rand() < mutation_prob
            mutation_point = randi(length(population(i, :)));
            mutated_population(i, :) = population(i, :) + randn() * 0.1;  % 基本变异：加上一个小的随机扰动
        end
    end
end