%%GA-BP
%%
%案例中 7个（列）指标（自变量） 1个（）列目标（因变量） 103个（行）样本
clear;clc;
close all;
warning off;
rng(6);

% 导入
data = xlsread('数据集.xlsx'); 
temp = randperm(103);
p_train = data(temp(1:80),1:7)'; % 训练集输入
t_train = data(temp(1:80),8)'; % 训练集输出
m = size(p_train,2);
p_test = data(temp(81:end),1:7)'; % 测试集输入
t_test = data(temp(81:end),8)'; % 测试集输出
n = size(p_test,2);

% 归一化处理
[pm_train,ps_int] = mapminmax(p_train,0,1); % 对训练集的自变量进行归一化
pm_test = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
[tm_train,ps_out] = mapminmax(t_train,0,1); % 对训练集的因变量进行归一化
tm_test = mapminmax('apply',t_test,ps_output); % 'apply'、'ps_input'对测试集因变量进行归一化

% 建立模型
node1 = 5;
net = newff(pm_train,tm_train,node1);
net.trainParam.epochs = 1000;
net.trainParam.goal   = 1e-6;
net.trainParam.lr     = 0.01;
ge_num = 50; % 遗传代数 需要测试优化 对模型结果影响较大
pp_num = 5;
s = size(pm_train,1)*node1 + size(tm_train,1)*node1 + size(tm_train,1) + node1;
bounds = ones(s,1)*[-1,1];

% 初始化种群
pc = [1e-6,1];
normgeomselect = 0.09;
artovr = 2;
nonunimutation = [2 ge_num 3];
intpp = initializega(pp_num,bounds,'gabpeval',[],pc);

% 优化
[bestpp,endpp,bpp,tre] = ga(bounds,'gabpeval',[],intpp,[pc,0],'maxgenterm',ge_num,...
                            'normgeomselect',normgeomselect,'artovr',artovr,...
                            'nonunimutation',nonunimutation);
[value,wgt1,b1,wgt2,b2] = gacod(bestpp); % 最优参数
net.IW{1,1} = wgt1; % 设置参数 
net.LW{2,1} = wgt2;
net.b{1}    = b1;
net.b{2}    = b2;
net.trainParam.showWindow = 1; % 打开训练窗口
net = train(net,pm_train,tm_train);
t_sim1 = sim(net,pm_train); % 仿真预测
t_sim2 = sim(net,pm_test);
tsim1 = mapminmax('reverse',t_sim1,ps_out); % 'reverse'对输出的因变量进行反归一化
tsim2 = mapminmax('reverse',t_sim2,ps_out);

wrong1 = sqrt(sum((tsim1 - t_train).^2)./m); % 得到均方根误差
wrong2 = sqrt(sum((tsim2 - t_test).^2)./n); 

% 绘制图像
figure
plot(1: m, t_train, 'm-*', 1: m, tsim1, 'c-o', 'LineWidth', 1);
xlabel('预测样本');ylabel('预测结果');
legend('真实值', '预测值');
string = ['训练集预测结果对比：均方根误差=',num2str(wrong1)];title(string);
xlim([1,m]);grid on;

figure
plot(1:n,t_test,'m-*',1:n,tsim2,'c-o','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
legend('真实值','预测值');
string = ['测试集预测结果对比：均方根误差=',num2str(wrong2)];title(string);
xlim([1,n]);grid on;

figure
plot(tre(:, 1), 1 ./ tre(:, 2), 'LineWidth', 1.5);
xlabel('迭代次数');ylabel('适应度值');
title('适应度变化曲线');
grid on;

size = 20;
color = 'c';
figure
scatter(t_train, tsim1, size, color);
hold on;
plot(xlim, ylim, '--k');
xlabel('训练集真实值');ylabel('训练集预测值');
xlim([min(t_train) max(t_train)]);ylim([min(tsim1) max(tsim1)]);
title('训练集真实值与预测值的对比图');

figure
scatter(t_test, tsim2, size, color);
hold on
plot(xlim, ylim, '--k');
xlabel('测试集真实值');ylabel('测试集预测值');
xlim([min(t_test) max(t_test)]);ylim([min(tsim2) max(tsim2)]);
title('测试集真实值与预测值的对比图');

% 指标结果
% 决定系数R2
R1 = 1-norm(t_train - tsim1)^2 / norm(t_train - mean(t_train))^2;
R2 = 1-norm(t_test  - tsim2)^2 / norm(t_test  - mean(t_test ))^2;
disp(['训练集数据的R2为：',num2str(R1)]);
disp(['测试集数据的R2为：',num2str(R2)]);

% 平均绝对误差MAE
mae1 = sum(abs(tsim1 - t_train))./m;
mae2 = sum(abs(tsim2 - t_test ))./n;
disp(['训练集数据的平均绝对误差为：',num2str(mae1)]);
disp(['测试集数据的平均绝对误差为：',num2str(mae2)]);

% 平均相对误差MBE
mbe1 = sum(tsim1 - t_train)./m;
mbe2 = sum(tsim2 - t_test )./n;
disp(['训练集数据的平均相对误差为：',num2str(mbe1)]);
disp(['测试集数据的平均相对误差为：',num2str(mbe2)]);

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
t_sim = sim(net,new_da); % 预测
tsim = mapminmax('reverse',t_sim,ps_output); % 'reverse'对输出的因变量进行反归一化
xlswrite('预测结果',tsim'); % 得到预测值
