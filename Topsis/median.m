% median函数，对中间型指标正向化
function A_ = median(A, weights)
    % 对每个中间型指标进行正向化
    jud = input('是否有中间型指标（0 or 1）：');
    if jud == 0
        A_ = A;
        return
    else
        [n, m] = size(A);
        a0 = input('请输入中间型指标的列数（一行矩阵）：');
        [k, l] = size(a0); % k = 1
        for i = 1:l
            x_best = input('请输入最佳的数值：');
            a01 = repmat(x_best, n, 1);
            y = a0(i);
            M = max(abs(a01(:, 1) - A(:, y)));
            A(:, y) = 1 - (abs(A(:, y) - x_best) / M);
        end
        A_ = A;
    end
end