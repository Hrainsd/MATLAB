%%层次分析法

%%
clc;clear
% 法一
A = input('请输入判断矩阵：');
size_A = size(A); %size_A = [n,n]
a = size_A(1);
product = prod(A,2);
result = product.^(1/a)/(sum(product.^(1/a)));
disp('各指标的权重为：')
disp(result)
char = eig(A);
char_max = max(char);
CI = (char_max - a)/(a - 1);
RI = [0.00 0.00 0.58 0.96 1.12 1.24 1.32 1.41 1.45];
CR = CI/RI(a) %RI<0.1,认为判断矩阵的一致性可以接受
if CR<0.1
    disp('判断矩阵的一致性可以接受')
else
    disp('判断矩阵的一致性不可以接受')
end

%%
% 法二
A = input('请输入判断矩阵：');
size_A = size(A); %size_A = [n,n]
a = size_A(1);
n=[];
for k = 1:a
    row = A(k,:);
    m = 1;
     for h = 1:a
         cell_value = row(h);
         m = m*cell_value;
     end
         n(k,1) =  m;
end
n_cf = n.^(1/a) ;
result1 = n_cf./sum(n_cf);
disp('各指标的权重为：')
disp(result1)
char = eig(A);
char_max = max(char);
CI = (char_max - a)/(a - 1);
RI = [0.00 0.00 0.58 0.96 1.12 1.24 1.32 1.41 1.45];
CR = CI/RI(a) %RI<0.1,认为判断矩阵的一致性可以接受
if CR<0.1
    disp('判断矩阵的一致性可以接受')
else
    disp('判断矩阵的一致性不可以接受')
end
