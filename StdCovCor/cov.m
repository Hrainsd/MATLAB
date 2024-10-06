function cov_ = cov(a,b)
% 求矩阵每行的协方差
[m,n] = size(a);
x_ = mean(a,2);
y_ = mean(b,2);
c_upper = 0;
for i = 1:m
    for j = 1:n
        c_upper = c_upper + (a(i,j) - x_(i))*(b(i,j) - y_(i));
    end
end
cov_ = c_upper/(n-1);
x2 = ['协方差为：',num2str(cov_)];
disp(x2); % 主对角线为自协方差 副对角线为协方差
end
