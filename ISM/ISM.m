%% 结构模型分析法ISM
%%
clc;
clear;

M = input('请输入邻接矩阵：');
[n0, m0] = size(M); % n0 = m0
M1 = M;

% 计算邻接矩阵的幂
for k = 1:n0
    M1 = M1 * M;
    M1(M1 > 0) = 1; % 将大于0的值置为1
    disp(['M的', num2str(k + 1), '次方为：']);
    disp(M1);
end

T = M1;

for k1 = 1:n0
    [n, m] = size(T);
    res = [];
    R = zeros(n, m);
    A = zeros(n, m);
    RA = zeros(n, m);

    % 计算 R 矩阵
    for i = 1:n
        for j = 1:m
            if T(i, j) == 1
                R(i, j) = j;
            end
        end
    end

    T = T'; % 转置 T 矩阵

    % 计算 A 矩阵
    for i = 1:n
        for j = 1:m
            if T(i, j) == 1
                A(i, j) = j;
            end
        end
    end

    % 计算 RA 矩阵
    for i = 1:n
        RA(i, :) = R(i, :) .* (R(i, :) == A(i, :)); % 逐元素比较并赋值
    end

    % 提取结果
    for i = 1:n
        if all(RA(i, :) == R(i, :))
            res = [res, R(i, R(i, :) > 0)]; % 收集结果
        end
    end

    res(res == 0) = [];
    res = unique(res);
    res = sort(res, 'descend');

    disp(['第', num2str(k1), '位要素为：']);
    disp(res);

    T = T'; % 转置回原矩阵
    [n1, m1] = size(res);

    % 更新 T 矩阵
    for i = 1:m1
        i_ = res(i);
        T(:, i_) = 0; % 清除列
        T(i_, :) = 0; % 清除行
    end

    if norm(T, inf) == 0
        break; % 如果 T 矩阵全为零则结束
    end
end
