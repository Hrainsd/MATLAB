% minmin函数，对极小型指标正向化
function A_ = minmin(A, weights)
    % 对每个极小型指标进行正向化
    jud = input('是否有极小型指标（0 or 1）：');
    if jud == 0
        A_ = A;
        return
    else
        [n, m] = size(A);
        a0 = input('请输入极小型指标的列数（一行矩阵）：');
        [k, l] = size(a0); % k = 1
        a01 = repmat(max(A), n, 1);
        for i = 1:l
            y = a0(i);
            A(:, y) = (weights(y) * (a01(:, y) - A(:, y))) / sum(weights);
        end
        A_ = A;
    end
end