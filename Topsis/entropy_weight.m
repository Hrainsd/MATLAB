% 熵权法计算指标权重的函数
function weights = entropy_weight(A)
    [n, m] = size(A);
    % 计算指标的相对熵值
    entropy_values = zeros(1, m);
    for j = 1:m
        p = A(:, j) / sum(A(:, j));
        entropy_values(j) = -sum(p .* log(p));
    end

    % 计算指标的权重
    weights = (1 - entropy_values) / sum(1 - entropy_values);
end