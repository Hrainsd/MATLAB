%%CNN
%%
%案例中 7个（列）指标（自变量） 1个（）列目标（因变量） 103个（行）样本
clear;clc;
close all;
warning off;
rng(6);

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

% 平铺数据 长度 宽度 通道数 样本数
pm_train =  double(reshape(pm_train, 7, 1, 1, m));
pm_test  =  double(reshape(pm_test , 7, 1, 1, n));
tm_train =  double(tm_train)';
tm_test  =  double(tm_test )';

% 建立网络
layers = [
    imageInputLayer([7, 1, 1])                         % 输入层 
    convolution2dLayer([3, 1], 16, 'Padding', 'same')  % 卷积核大小为3*1 特征图16张
    batchNormalizationLayer                            % 批归一化层
    reluLayer                                          % Relu激活层
    maxPooling2dLayer([2, 1], 'Stride', [1, 1])        % 最大池化层 池化窗口为[2, 1] 步长为[1, 1]
    convolution2dLayer([3, 1], 32, 'Padding', 'same')  % 卷积核大小为3*1 特征图32张
    batchNormalizationLayer                            % 批归一化层
    reluLayer                                          % Relu激活层
    dropoutLayer(0.1)                                  % 丢弃层
    fullyConnectedLayer(1)                             % 全连接层 一个输出，值为1
    regressionLayer];                                  % 回归层

% 设置参数
options = trainingOptions('sgdm', ...      % SGDM 梯度下降算法
    'MiniBatchSize', 32, ...               % 批大小 每次有32个样本训练
    'MaxEpochs', 600, ...                 % 最大训练次数 1200
    'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子
    'LearnRateDropPeriod', 800, ...        % 800次训练后 学习率为初始学习率*学习率下降因子
    'Shuffle', 'every-epoch', ...          % 打乱数据集
    'Plots', 'training-progress', ...      % 画曲线
    'Verbose', false);

% 仿真预测
net = trainNetwork(pm_train, tm_train, layers, options);
t_sim1 = predict(net, pm_train);
t_sim2 = predict(net, pm_test );
tsim1 = mapminmax('reverse',t_sim1,ps_output); % 'reverse'对输出的因变量进行反归一化
tsim2 = mapminmax('reverse',t_sim2,ps_output);

wrong1 = sqrt(sum((tsim1' - t_train).^2)./m); % 得到均方误差
wrong2 = sqrt(sum((tsim2' - t_test ).^2)./n); 

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

scale = 20;
color = 'c';
figure
scatter(t_train, tsim1, scale, color);
hold on;
plot(xlim, ylim, '--k');
xlabel('训练集真实值');ylabel('训练集预测值');
xlim([min(t_train) max(t_train)]);ylim([min(tsim1) max(tsim1)]);
title('训练集真实值与预测值的对比图');

figure
scatter(t_test, tsim2, scale, color);
hold on;
plot(xlim, ylim, '--k');
xlabel('测试集真实值');ylabel('测试集预测值');
xlim([min(t_test) max(t_test)]);ylim([min(tsim2) max(tsim2)]);
title('测试集真实值与预测值的对比图');

analyzeNetwork(layers);
% 指标结果
% 平均相对误差
mbe1 = sum(tsim1' - t_train)./m;
mbe2 = sum(tsim2' - t_test )./n;
disp(['训练集数据的平均相对误差为：',num2str(mbe1)]);
disp(['测试集数据的平均相对误差为：',num2str(mbe2)]);

% 平均绝对误差MAE
mae1 = sum(abs(tsim1' - t_train))./m;
mae2 = sum(abs(tsim2' - t_test))./n;
disp(['训练集数据的平均绝对误差为：',num2str(mae1)]);
disp(['测试集数据的平均绝对误差为：',num2str(mae2)]);

% 决定系数R2
R1 = 1-norm(t_train - tsim1')^2 / norm(t_train - mean(t_train))^2;
R2 = 1-norm(t_test  - tsim2')^2 / norm(t_test  - mean(t_test ))^2;
disp(['训练集数据的R2为：',num2str(R1)]);
disp(['测试集数据的R2为：',num2str(R2)]);

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

new_da  = xlsread('需要预测的数据.xlsx');
clear size;
[k1,k2] = size(new_da);
new_da  = new_da';
new_da  = mapminmax('apply',new_da,ps_input);
new_da  = double(reshape(new_da, 7, 1, 1, k1)); % 平铺数据
t_sim   = predict(net,new_da); % 预测
tsim    = mapminmax('reverse',t_sim,ps_output); % 'reverse'对输出的因变量进行反归一化
xlswrite('预测结果',tsim); % 得到预测值
