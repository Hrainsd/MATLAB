%灰色关联分析法
%%
clc;clear

a = input('请输入行为样本，列为指标的矩阵：');

% [1988	386	839	763
% 2061	408	846	808
% 2335	422	960	953
% 2750	482	1258	1010
% 3356	511	1577	1268
% 3806	561	1893	1352]

[m,n] = size(a);
%标准化处理
mn = mean(a);
a1 = zeros();
for k = 1:m
    for l = 1:n
    a1(k,l) = a(k,l)./mn(1,l); 
    end
end
a1
%确定母序列
%计算两级最小差和两级最大差
%若母序列为每个样本的各指标最大值组成的矩阵：
%c = max(a')
%c'

%%
% 假定母序列为第一列组成的矩阵;子序列为其它列组成的矩阵
mot = a1(:,1);
chi = a1(:,2:n); % chi 为[m,n-1]
pro_abs = abs(chi-mot);
min_pro = min(min(pro_abs));
max_pro = max(max(pro_abs));
ro = 0.5;
y = (min_pro + ro*max_pro)./(pro_abs +ro*max_pro);
result = mean(y) % 得出灰色关联系数
weight = result/sum(result)
sorce = chi*weight';
standard_sorce = sorce/sum(sorce)
[sorted_S,index] = sort(standard_sorce ,'descend') % 得分降序排列，index为相应得分对应样本的位置索引（该案例第六个样本的得分最高）
x_plot1 = 1:1:n-1;
x_plot2 = 1:1:m;
subplot(1,2,1);
plot(x_plot1,result,'-b');shading interp;
xlabel('指标');ylabel('灰色关联度');title('关联度曲线图');xticks([0:1:n]); % xticklabels({'指标1','指标2','指标3'});
grid on;
subplot(1,2,2);
plot(x_plot2,standard_sorce,'-c');shading interp;
xlabel('样本对象');ylabel('得分');title('得分曲线图');xticks([0:1:m+1]); % xticklabels({'样本1','样本2','样本3','样本4','样本5','样本6'});
grid on; 
