%%CNN
%%
%案例中 12个（列）指标（自变量） 1个（）列目标（因变量） 357个（行）样本
clear;clc;
close all;
warning off;

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
pm_test  = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
tm_train =  categorical(t_train)';
tm_test  =  categorical(t_test )';

% 平铺数据
pm_train =  double(reshape(pm_train, 12, 1, 1, m));
pm_test  =  double(reshape(pm_test , 12, 1, 1, n));

% 建立网络
layers = [
    imageInputLayer([12, 1, 1])                        % 输入层 
    convolution2dLayer([2, 1], 16, 'Padding', 'same')  % 卷积核大小为2*1 特征图16张
    batchNormalizationLayer                            % 批归一化层
    reluLayer                                          % Relu激活层
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])        % 最大池化层 池化窗口为[2, 1] 步长为[2, 1]
    convolution2dLayer([2, 1], 32, 'Padding', 'same')  % 卷积核大小为2*1 特征图32张
    batchNormalizationLayer                            % 批归一化层
    reluLayer                                          % Relu激活层
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])        % 最大池化层 池化窗口为[2, 1] 步长为[2, 1]
    fullyConnectedLayer(4)                             % 全连接层 四个输出，值为4
    softmaxLayer                                       % 损失函数层
    classificationLayer];                              % 分类层

% 设置参数
options = trainingOptions('adam', ...      % ADAM 梯度下降算法
    'MaxEpochs', 500, ...                 % 最大训练次数 1200
    'InitialLearnRate', 1e-3, ...          % 初始学习率为0.01
    'L2Regularization', 1e-4, ...          % L2 正则化参数
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子
    'LearnRateDropPeriod', 400, ...        % 800次训练后 学习率为初始学习率*学习率下降因子
    'Shuffle', 'every-epoch', ...          % 打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 画曲线
    'Verbose', false);

% 仿真预测
net = trainNetwork(pm_train, tm_train, layers, options);
t_sim1 = predict(net, pm_train);
t_sim2 = predict(net, pm_test );
tsim1 = vec2ind(t_sim1');
tsim2 = vec2ind(t_sim2');
[t_train, id_1] = sort(t_train);
[t_test , id_2] = sort(t_test );
tsim1 = tsim1(id_1);
tsim2 = tsim2(id_2);

wrong1 = sum((t_train == tsim1)) /m *100; % 得到均方误差
wrong2 = sum((t_test  == tsim2)) /n *100; 

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

analyzeNetwork(layers);

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

new_da  = xlsread('需要预测的数据.xlsx');
[k1,k2] = size(new_da);
new_da  = new_da';
new_da  = mapminmax('apply',new_da,ps_input);
new_da  = double(reshape(new_da, 12, 1, 1, k1)); % 平铺数据
t_sim   = predict(net,new_da); % 预测
tsim    = vec2ind(t_sim');
xlswrite('预测结果',tsim'); % 得到预测值
