function Standard = std(a)
% 求矩阵每行的标准差
[m,n] = size(a);
standard_ = 0;
x_ = mean(a,2);
for i = 1:m
    for j = 1:n
        standard_ = standard_ + (a(i,j) - x_(i))^2;
    end
end
Standard = sqrt(standard_/(n-1)); 
x = ['标准差为：',num2str(Standard)];
disp(x);
end
