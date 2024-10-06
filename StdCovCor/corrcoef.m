function r = corrcoef(a,b)
% 求矩阵每行的相关系数
[m,n] = size(a);
x_ = mean(a,2);
y_ = mean(b,2);
r_upper = 0;
r_lower1 = 0;
r_lower2 = 0;
for i = 1:m
    for j = 1:n
        r_upper = r_upper + (a(i,j) - x_(i))*(b(i,j) - y_(i));
        r_lower1 = r_lower1 + (a(i,j) - x_(i))^2;
        r_lower2 = r_lower2 + (b(i,j) - y_(i))^2;
    end
end
r = r_upper/sqrt(r_lower1*r_lower2);
x1 = ['相关系数为：',num2str(r)];
disp(x1); % 主对角线为自相关系数 副对角线为相关系数
end

