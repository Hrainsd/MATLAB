%相对偏差模糊矩阵评价法
%%
clc;clear

% 变异系数法（权重越大，该指标分辨信息越丰富）
A = input('请输入矩阵（列为指标，行为方案）：')
[m,n] = size(A);
x_bar = zeros(1,n); %平均数
var_2 = zeros(1,n); %方差
for k = 1:n
    x_bar(k) = (1/m)*(sum(A(:,k))) %x_bar = mean(A)对列求方差
end
for q = 1:n
    med = 0;
    for b = 1:m
        med = med + (A(b,q)-x_bar(q))^2;
    end
        var_2(q) = (1/(m-1))*med;
end %var_2 = std(A)
v = sqrt(var_2)./abs(x_bar); 
w = v/sum(v) %求权重
%%
% 构建虚拟方案u = (u1,u2,u3...un)
%极大型指标(效益型)ui = max{a(i,j)};极小型指标(成本型)ui = min{a(i,j)}
MAX_A = max(A) ;
MIN_A = min(A);
D = MAX_A - MIN_A;
M_1 = max(A(:,1));
M_2 = min(A(:,2:6));
M_3 = max(A(:,7));
u = [M_1,M_2,M_3];
r = zeros(m,n);
for i = 1:m
    for j = 1:n
        R(i,j) = abs(A(i,j)-u(j))/D(j);
    end
end
F = R*w';
F' %F的值越小，方案越好
