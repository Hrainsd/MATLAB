% clc;clear;
X = input('请输入n×m的矩阵（n个样本m个指标）：');
[n,m] = size(X);
JDE = 0;
for i =1:n
    for j =1:m
        if X(i,j) < 0
            JDE = 1;
            break
        else
            JDE = 0;
        end
    end
end
max_X = max(X);min_X = min(X);
Y = zeros(n,m);
if JDE == 0
    for i =1:n
        for j = 1:m
            Y(i,j) = 0.998*((X(i,j) - min_X(j))/(max_X(j) - min_X(j))) + 0.002;
        end
    end
else
    for i =1:n
        for j = 1:m
            Y(i,j) = 0.998*((min_X(j) - X(i,j))/(max_X(j) - min_X(j))) + 0.002;
        end
    end
end
P0 = sum(Y);P0 = repmat(P0,n,1);
P = Y./P0;
E = zeros(1,m);
for j =1:m
    for i =1:n
        if P(i,j) == 0
            E(j) = 0;
            break
        else
            E(j) = E(j) + P(i,j)*log(P(i,j))/-log(n);
        end
    end
end
W = (1 - E)./(m - sum(E))
W1 = repmat(W,n,1);
S = sum(W1.*X,2);