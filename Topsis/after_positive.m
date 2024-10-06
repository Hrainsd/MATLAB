% after_positive函数，计算得分
function [sorted_S, index] = after_positive(A)
    [n, m] = size(A); 
    a = zeros(n, m);
    for i = 1:n
        for j = 1:m
            a(i, j) = A(i, j) / sqrt(sum(A(:, j).^2));
        end
    end
    a1 = max(a);
    a2 = min(a);
    a_max = zeros(n, 1);
    a_min = zeros(n, 1);
    for i = 1:n
        for j = 1:m
            a_max(i, 1) = a_max(i, 1) + (a1(j) - a(i, j))^2;
        end
        a_max(i, 1) = sqrt(a_max(i, 1));
    end
    for i = 1:n
        for j = 1:m
            a_min(i, 1) = a_min(i, 1) + (a2(j) - a(i, j))^2;
        end
        a_min(i, 1) = sqrt(a_min(i, 1));
    end
    score = a_min ./ (a_max + a_min);
    score_standard = score / sum(score);
    [sorted_S, index] = sort(score_standard, 'descend'); % 降序排列得分，index输出索引值
end