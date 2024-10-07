%%MultipleOutputsBP
%%
%案例中 28个（列）指标（自变量） 3个（）列目标（因变量） 719个（行）样本
clear;clc;
close all;
warning off;
rng(6);

% 导入
res = xlsread('多输出数据集.xlsx'); 
temp = randperm(719);
p_train = res(temp(1:500),1:28)'; % 训练集输入
t_train = res(temp(1:500),29:31)'; % 训练集输出
m = size(p_train,2);
p_test = res(temp(501:end),1:28)'; % 测试集输入
t_test = res(temp(501:end),29:31)'; % 测试集输出
n = size(p_test,2);

% 归一化处理
[pm_train,ps_input] = mapminmax(p_train,0,1); % 对训练集的自变量进行归一化
pm_test = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
[tm_train,ps_output] = mapminmax(t_train,0,1); % 对训练集的因变量进行归一化
tm_test = mapminmax('apply',t_test,ps_output); % 'apply'、'ps_input'对测试集因变量进行归一化

% 搭建网络
net = newff(pm_train,tm_train,14);
net.trainParam.epochs = 1000;   % 迭代次数1000
net.trainParam.goal = 1e-6;     % 误差阈值1e-6    
net.trainParam.lr = 0.01;       % 学习率0.01
net.trainFcn = 'trainlm';

net = train(net,pm_train,tm_train);
t_sim1 = sim(net,pm_train);
t_sim2 = sim(net,pm_test);
tsim1 = mapminmax('reverse',t_sim1,ps_output); % 'reverse'对输出的因变量进行反归一化
tsim2 = mapminmax('reverse',t_sim2,ps_output);

for i = 1: 3
wrong1(i, :) = sqrt(sum((tsim1(i,:) - t_train(i,:)).^2)./m); % 得到均方误差
wrong2(i, :) = sqrt(sum((tsim2(i,:) - t_test(i,:) ).^2)./n);
disp(['第',num2str(i),'个因变量预测结果：']);

% 可视化
figure 
subplot(1, 2, 1)
plot(1:m,t_train(i,:),'m-*',1:m,tsim1(i,:),'c-o','LineWidth',0.5);
xlabel('预测样本');ylabel('预测结果');
string = ['训练集预测结果对比：RMSE = ',num2str(wrong1(i,:))];title(string);
xlim([1,m]);legend('真实值','预测值');grid on;

subplot(1, 2, 2)
plot(1:n,t_test(i,:),'m-*',1:n,tsim2(i,:),'c-o','LineWidth',0.5);
xlabel('预测样本');ylabel('预测结果');
string = ['测试集预测结果对比：RMSE = ',num2str(wrong2(i,:))];title(string);
xlim([1,n]);legend('真实值','预测值');grid on;

% 指标结果
% 平均相对误差
mbe1(i,:) = sum(tsim1(i,:) - t_train(i,:))./m;
mbe2(i,:) = sum(tsim2(i,:) - t_test(i,:))./n;
disp(['训练集数据的平均相对误差为：',num2str(mbe1(i,:))]);
disp(['测试集数据的平均相对误差为：',num2str(mbe2(i,:))]);

% 平均绝对误差MAE
mae1(i,:) = sum(abs(tsim1(i,:) - t_train(i,:)))./m;
mae2(i,:) = sum(abs(tsim2(i,:) - t_test(i,:)))./n;
disp(['训练集数据的平均绝对误差为：',num2str(mae1(i,:))]);
disp(['测试集数据的平均绝对误差为：',num2str(mae2(i,:))]);

% 决定系数R2
R1(i,:) = 1-norm(t_train(i,:) - tsim1(i,:))^2 / norm(t_train(i,:) - mean(t_train(i,:)))^2;
R2(i,:) = 1-norm(t_test(i,:) -  tsim2(i,:))^2 / norm(t_test(i,:)  -  mean(t_test(i,:)))^2;
disp(['训练集数据的R2为：',num2str(R1(i,:))]);
disp(['测试集数据的R2为：',num2str(R2(i,:))]);
end

%% 

% 读取新数据，并预测保存
% new_da = xlsread('新数据文件名');
%举例用数据集后80个样本进行数据预测 

t_test3 = res(501:end,29:31)';
new_da = res(501:end,1:28);
new_da = new_da';
new_da = mapminmax('apply',new_da,ps_input);
t_sim3 = sim(net,new_da);
tsim3 = mapminmax('reverse',t_sim3,ps_output);
xlswrite('预测结果',tsim3'); % 得到一行预测值

for i = 1: 3
wrong3(i, :) = sqrt(sum((tsim3(i,:) - t_test3(i,:)).^2)./n); % 得到均方误差
disp(['第',num2str(i),'个因变量预测结果：']);

% 可视化

figure 
plot(1:n,t_test3(i,:),'m-*',1:n,tsim3(i,:),'c-o','LineWidth',0.5);
xlabel('预测样本');ylabel('预测结果');
string = ['预测结果对比：RMSE = ',num2str(wrong3(i,:))];title(string);
xlim([1,n]);legend('真实值','预测值');grid on;

% 指标结果
% 平均相对误差

mbe3(i,:) = sum(tsim3(i,:) - t_test3(i,:))./n;
disp(['测试集数据的平均相对误差为：',num2str(mbe3(i,:))]);

% 平均绝对误差MAE

mae3(i,:) = sum(abs(tsim3(i,:) - t_test3(i,:)))./n;
disp(['测试集数据的平均绝对误差为：',num2str(mae3(i,:))]);

% 决定系数R2

R3(i,:) = 1-norm(t_test3(i,:) -  tsim3(i,:))^2 / norm(t_test3(i,:)  -  mean(t_test3(i,:)))^2;
disp(['测试集数据的R2为：',num2str(R3(i,:))]);
end

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
t_sim =  sim(net,new_da); % 预测
tsim  =  mapminmax('reverse',t_sim,ps_output); % 'reverse'对输出的因变量进行反归一化
xlswrite('预测结果',tsim'); % 得到预测值
