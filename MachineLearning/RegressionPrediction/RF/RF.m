%%RF
%%
%案例中 7个（列）指标（自变量） 1个（）列目标（因变量） 103个（行）样本
clear;clc;
close all;
warning off;
rng(12);

% 导入
res = xlsread('数据集.xlsx'); 
temp = randperm(103);
p_train = res(temp(1:80),1:7)'; % 训练集输入
t_train = res(temp(1:80),8)'; % 训练集输出
m = size(p_train,2);
p_test = res(temp(81:end),1:7)'; % 测试集输入
t_test = res(temp(81:end),8)'; % 测试集输出
n = size(p_test,2);

% 归一化处理
[pm_train,ps_input] = mapminmax(p_train,0,1); % 对训练集的自变量进行归一化
pm_test = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
[tm_train,ps_output] = mapminmax(t_train,0,1); % 对训练集的因变量进行归一化
tm_test = mapminmax('apply',t_test,ps_output); % 'apply'、'ps_input'对测试集因变量进行归一化

% 建立模型
pm_train = pm_train';pm_test = pm_test';
tm_train = tm_train';tm_test = tm_test';
d_tree   = 150; % 决策树
min_leaf = 0.6; % 最小叶子数
oob_pdn  = 'on'; % 误差图
oob_pdn_imp = 'on'; % 特征重要性
med = 'regression'; % 回归预测
net = TreeBagger(d_tree,pm_train,tm_train,'OOBPrediction',oob_pdn,...
      'OOBPredictorImportance',oob_pdn_imp,'Method',med,'minLeaf',min_leaf);
imp = net.OOBPermutedPredictorDeltaError;

t_sim1 = predict(net,pm_train); % 预测
t_sim2 = predict(net,pm_test);
tsim1 = mapminmax('reverse',t_sim1,ps_output); % 'reverse'对输出的因变量进行反归一化
tsim2 = mapminmax('reverse',t_sim2,ps_output);

wrong1 = sqrt(sum((tsim1' - t_train).^2)./m); % 得到均方根误差
wrong2 = sqrt(sum((tsim2' - t_test).^2)./n); 

% 可视化
figure 
plot(1:m,t_train,'m-*',1:m,tsim1,'c-o','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
string = ['训练集预测结果对比：RMSE = ',num2str(wrong1)];title(string);
xlim([1,m]);legend('真实值','预测值');grid on;

figure
plot(1:n,t_test,'m-*',1:n,tsim2,'c-o','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
string = ['测试集预测结果对比：RMSE = ',num2str(wrong2)];title(string);
xlim([1,n]);legend('真实值','预测值');grid on;

figure
plot(1:d_tree,oobError(net),'-k','LineWidth',1);
xlabel('决策树的个数');ylabel('误差值');
xlim([1,d_tree]);legend('误差曲线图');grid on;

figure
bar(imp);
xlabel('特征指标');ylabel('重要性');

% 指标结果
% 平均相对误差MBE
mbe1 = sum(tsim1' - t_train)./m;
mbe2 = sum(tsim2' - t_test)./n;
disp(['训练集数据的平均相对误差为：',num2str(mbe1)]);
disp(['测试集数据的平均相对误差为：',num2str(mbe2)]);

% 平均绝对误差MAE
mae1 = sum(abs(tsim1' - t_train))./m;
mae2 = sum(abs(tsim2' - t_test))./n;
disp(['训练集数据的平均绝对误差为：',num2str(mae1)]);
disp(['测试集数据的平均绝对误差为：',num2str(mae2)]);

% 决定系数R2
R1 = 1-norm(t_train - tsim1')^2 / norm(t_train - mean(t_train))^2;
R2 = 1-norm(t_test - tsim2')^2 / norm(t_test - mean(t_test))^2;
disp(['训练集数据的R2为：',num2str(R1)]);
disp(['测试集数据的R2为：',num2str(R2)]);

disp('********************')

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
new_da = new_da';
new_da = mapminmax('apply',new_da,ps_input);
new_da = new_da';
t_sim  = predict(net,new_da); % 预测
tsim   = mapminmax('reverse',t_sim,ps_output); % 'reverse'对输出的因变量进行反归一化
xlswrite('预测结果',tsim); % 得到预测值
