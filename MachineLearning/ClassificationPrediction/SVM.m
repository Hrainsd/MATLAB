%%SVM
%%
%案例中 12个（列）指标（自变量） 1个（）列目标（因变量） 357个（行）样本
clear;clc;
close all;
warning off;
rng(12);

% 导入
res = xlsread('分类预测数据.xlsx'); 
temp = randperm(357);
p_train = res(temp(1:240),1:12)'; % 训练集输入
t_train = res(temp(1:240),13)'; % 训练集输出
m = size(p_train,2);
p_test = res(temp(241:end),1:12)'; % 测试集输入
t_test = res(temp(241:end),13)'; % 测试集输出
n = size(p_test,2);

% 归一化处理
[pm_train,ps_input] = mapminmax(p_train,0,1); % 对训练集的自变量进行归一化
pm_test = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
tm_train = t_train;
tm_test  = t_test ;
pm_train = pm_train';pm_test = pm_test';
tm_train = tm_train';tm_test = tm_test';

% 建立模型
c = 10.0; % 惩罚因子
g = 0.01; % 径向基函数参数
pst = [' -t 2',' -c ',num2str(c),' -g ',num2str(g)];
pat = svmtrain(tm_train,pm_train,pst);
tsim1 = svmpredict(tm_train,pm_train,pat); % 预测
tsim2 = svmpredict(tm_test,pm_test,pat);
[t_train, id_1] = sort(t_train);
[t_test , id_2] = sort(t_test );
tsim1 = tsim1(id_1);
tsim2 = tsim2(id_2);

wrong1 = sum((t_train == tsim1')) /m *100; % 得到均方误差
wrong2 = sum((t_test  == tsim2')) /n *100; 

% 可视化
figure 
plot(1:m,t_train,'m-*',1:m,tsim1,'c-o','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
string = ['训练集预测结果对比：准确率 = ',num2str(wrong1),'%'];title(string);
xlim([1,m]);legend('真实值','预测值');grid on;

figure
plot(1:n,t_test,'m-*',1:n,tsim2,'c-o','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
string = ['测试集预测结果对比：准确率 = ',num2str(wrong2),'%'];title(string);
xlim([1,n]);legend('真实值','预测值');grid on;

figure % 混淆矩阵
ct               = confusionchart(t_train, tsim1);
ct.Title         = 'Confusion Matrix for Train Data';
ct.RowSummary    = 'row-normalized';
ct.ColumnSummary = 'column-normalized';

figure
ct               = confusionchart(t_test, tsim2);
ct.Title         = 'Confusion Matrix for Test Data';
ct.RowSummary    = 'row-normalized';
ct.ColumnSummary = 'column-normalized';

%%
% % 读取新数据，并预测保存
% new_da = xlsread('新数据文件名');
% new_da = new_da';
% new_da = mapminmax('apply',new_da,ps_input);
% t_sim3 = sim(net,new_da);
% tsim3 = mapminmax('reverse',t_sim3,ps_output);
% xlswrite('保存文件名',tsim3'); % 得到一列预测值

%%
% 读取新数据，并预测保存
% new_da = xlsread('新数据文件名');

new_da = xlsread('需要预测的数据.xlsx');
[k1,k2] = size(new_da);
new_da = new_da';
new_da = mapminmax('apply',new_da,ps_input);
new_da = new_da';
yin_bian_liang = zeros(k1,1);
tsim = svmpredict(yin_bian_liang,new_da,pat); % 预测
xlswrite('预测结果',tsim); % 得到预测值
