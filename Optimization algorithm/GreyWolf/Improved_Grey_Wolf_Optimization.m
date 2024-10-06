function [best_solution_min, best_fitness_min, best_solution_max, best_fitness_max] = Improved_Grey_Wolf_Optimization(obj_func, dim, lb, ub, max_iter, pack_size)
    % Inner function to initialize positions with Tent chaotic map
    function initial_positions = initialize_positions_with_tent_chaotic(pack_size, dim, lb, ub, optimizationType)
        initial_positions = zeros(pack_size, dim);

        for i = 1:pack_size
            tent_chaos = tent_map_chaos_generator(dim);
            initial_positions(i, :) = lb + (ub - lb) * tent_chaos;
        end

        if strcmp(optimizationType, 'max')
            initial_positions = flipud(initial_positions);
        end
    end

    % Inner function to generate Tent chaotic map
    function chaos_sequence = tent_map_chaos_generator(sequence_length)
        chaos_sequence = zeros(1, sequence_length);
        x = rand(); % Initial value

        for i = 1:sequence_length
            x = tent_map(x);
            chaos_sequence(i) = x;
        end
    end

    % Inner function to apply Tent map
    function next_value = tent_map(current_value)
        if current_value <= 0.5
            next_value = 2 * current_value;
        else
            next_value = 2 * (1 - current_value);
        end
    end

    % Inner function to calculate adaptive hunting weight
    function hunting_weight = adaptive_hunting_weight(iter, max_iter)
        alpha = 2; % Initial value
        beta  = 0.1; % Initial value
        gamma = 10; % Scaling factor

        hunting_weight = alpha * exp(-beta * iter / max_iter) + gamma;
    end

    % Inner function to calculate adaptive control parameter
    function control_parameter = adaptive_control_parameter(iter, max_iter)
        control_parameter = 2 - iter * (2 / max_iter); % Improved control parameter
    end

    % Main loop of improved grey wolf optimization algorithm
    function [best_solution, best_fitness] = improved_grey_wolf_algorithm(optimizationType)
        alpha_pos = initialize_positions_with_tent_chaotic(1, dim, lb, ub, optimizationType);
        beta_pos  = initialize_positions_with_tent_chaotic(1, dim, lb, ub, optimizationType);
        delta_pos = initialize_positions_with_tent_chaotic(1, dim, lb, ub, optimizationType);

        alpha_score = feval(obj_func, alpha_pos);
        beta_score  = feval(obj_func, beta_pos);
        delta_score = feval(obj_func, delta_pos);

        positions = initialize_positions_with_tent_chaotic(pack_size, dim, lb, ub, optimizationType);

        for iter = 1:max_iter
            a = adaptive_control_parameter(iter, max_iter); % Improved control parameter

            hunting_weight = adaptive_hunting_weight(iter, max_iter); % Adaptive hunting weight

            for i = 1:pack_size
                fitness = feval(obj_func, positions(i, :));

                if (strcmp(optimizationType, 'min') && fitness < alpha_score) || (strcmp(optimizationType, 'max') && fitness > alpha_score)
                    alpha_score = fitness;
                    alpha_pos   = positions(i, :);
                end

                if (strcmp(optimizationType, 'min') && fitness > alpha_score && fitness < beta_score) || (strcmp(optimizationType, 'max') && fitness < alpha_score && fitness > beta_score)
                    beta_score = fitness;
                    beta_pos   = positions(i, :);
                end

                if (strcmp(optimizationType, 'min') && fitness > alpha_score && fitness > beta_score && fitness < delta_score) || (strcmp(optimizationType, 'max') && fitness < alpha_score && fitness < beta_score && fitness > delta_score)
                    delta_score = fitness;
                    delta_pos   = positions(i, :);
                end
            end

            for i = 1:pack_size
                r1 = rand();
                r2 = rand();

                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;

                D_alpha = abs(C1 * alpha_pos - positions(i, :));
                X1      = alpha_pos - hunting_weight * A1 * D_alpha; % Improved position update

                r1 = rand();
                r2 = rand();

                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;

                D_beta = abs(C2 * beta_pos - positions(i, :));
                X2     = beta_pos - hunting_weight * A2 * D_beta; % Improved position update

                r1 = rand();
                r2 = rand();

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * delta_pos - positions(i, :));
                X3 = delta_pos - hunting_weight * A3 * D_delta; % Improved position update

                positions(i, :) = (X1 + X2 + X3) / 3;

                positions(i, :) = max(positions(i, :), lb);
                positions(i, :) = min(positions(i, :), ub);
            end
        end

        best_solution = alpha_pos;
        best_fitness  = alpha_score;
    end

    % Call the algorithm for both minimization and maximization
    [best_solution_min, best_fitness_min] = improved_grey_wolf_algorithm('min');
    [best_solution_max, best_fitness_max] = improved_grey_wolf_algorithm('max');
end