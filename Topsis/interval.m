% interval函数，对区间型指标正向化
function A_ = interval(A, weights)
    % 对每个区间型指标进行正向化
    jud = input('是否有区间型指标（0 or 1）：');
    if jud == 0
        A_ = A;
        return
    else
        [n, m] = size(A);
        a0 = input('请输入区间型指标的列数（一行矩阵）：');
        [k, l] = size(a0); % k = 1
        a01 = max(A);
        a02 = min(A);
        M = zeros(k, l);
        for p = 1:l
            y = a0(p);
            a03 = input('请以[a,b]形式输入最佳区间：');
            con_a = a03(1); con_b = a03(2);
            M(k, p) = max(con_a - a02(y), a01(y) - con_b);
            for j = 1:l
                for i = 1:n
                    if A(i, y) < con_a
                        A(i, y) = 1 - ((con_a - A(i, y)) / M(k, p));
                    elseif A(i, y) >= con_a && A(i, y) <= con_b
                        A(i, y) = 1;
                    else
                        A(i, y) = 1 - ((A(i, y) - con_b) / M(k, p));
                    end
                end
            end
        end
        A_ = A;
    end
end