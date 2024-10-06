function new_population = adaptive_genetic_algorithm(population, fitness, crossover_prob, mutation_prob, num_selections, optimizationType)
    cumulative_prob = cumsum(fitness) / sum(fitness);
    selected_indices = zeros(1, num_selections);

    for i = 1:num_selections
        rand_num = rand();
        selected_indices(i) = find(cumulative_prob >= rand_num, 1, 'first');
    end
    
    crossovered_population = crossover(population(selected_indices, :), crossover_prob);
    mutated_population = adaptive_mutation(crossovered_population, mutation_prob, fitness);
    
    new_population = mutated_population;
    
    % 根据优化类型选择最大化或最小化
    if strcmpi(optimizationType, 'max')
        [~, idx_max] = max(fitness);
        new_population(end) = population(idx_max, :);
    elseif strcmpi(optimizationType, 'min')
        [~, idx_min] = min(fitness);
        new_population(1) = population(idx_min, :);
    else
        error('Invalid optimizationType. Use ''min'' or ''max''.');
    end
end

function crossovered_population = crossover(parents, crossover_prob)
    crossovered_population = zeros(size(parents));
    
    for i = 1:2:size(parents, 1)
        if rand() < crossover_prob
            crossover_point = randi(size(parents, 2));
            crossovered_population(i, :) = [parents(i, 1:crossover_point), parents(i+1, crossover_point+1:end)];
            crossovered_population(i+1, :) = [parents(i+1, 1:crossover_point), parents(i, crossover_point+1:end)];
        else
            crossovered_population(i, :) = parents(i, :);
            crossovered_population(i+1, :) = parents(i+1, :);
        end
    end
end

function mutated_population = adaptive_mutation(population, mutation_prob, fitness)
    % 自适应变异概率，根据适应度进行动态调整
    normalized_fitness = fitness / sum(fitness);
    mutation_prob = mutation_prob * (1 - normalized_fitness);

    mutated_population = population;
    
    for i = 1:length(population)
        if rand() < mutation_prob(i)
            mutation_point = randi(length(population(i, :)));
            mutated_population(i, :) = population(i, :) + randn() * 0.1;  % 基本变异：加上一个小的随机扰动
        end
    end
end