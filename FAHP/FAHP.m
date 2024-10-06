%模糊综合评判法
%%
clc;clear

D = input('请输入专家评价结果矩阵：')
P = input('请输入评审各要素的权系数行向量：')
E = input('请输入评价基准相应的价值量：')
sz = size(D);
a = sz(1);
for k = 1:a
    d = sum(D(k,:));
    R(k,:) = D(k,:)./d;
end
R
N = P*R*E'
