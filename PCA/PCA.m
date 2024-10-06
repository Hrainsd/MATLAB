%%
clc;clear;

%%
A = input('请输入矩阵（m个样本n个指标的m×n矩阵）：');
[m,n] = size(A);
B = input('请输入累计贡献率要求值（例如：0.8）：');
A1 = corrcoef(A);
[TZXL0,TZZ0] = eig(A1); % 特征向量，特征值
lmd0 = diag(TZZ0);
[lmd,index] = sort(lmd0,'descend');
TZXL = TZXL0(:,index);
ctn = zeros(n,1);
for i = 1:n
    ctn(i) = lmd(i)/sum(lmd);
end
cum_ctn = cumsum(lmd)/sum(lmd);
disp('贡献率为：');disp(ctn);
disp('累计贡献率为：');disp(cum_ctn);
disp('相对应的特征向量矩阵为：');disp(TZXL);
num = 1;
for i =1:n
    if cum_ctn(i) < B
        num = num + 1;
    else
        break
    end
end
disp(['只需要取前',num2str(num),'个主成分进行分析'])

disp('满足累计贡献率要求的主成分值为：');
for i = 1:num
    disp(['主成分', num2str(i), '的值为：']);
    disp(A * TZXL(:,i));
end
