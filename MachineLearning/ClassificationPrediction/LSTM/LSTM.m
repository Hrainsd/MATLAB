%%LSTM
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
for i = 1:m
    pmm_train{i,1} = pm_train(:,:,1,i);
end
for j = 1:n
    pmm_test{j,1}  = pm_test(:,:,1,j );
end

layers = [
         sequenceInputLayer(12)            % 输入层 
         lstmLayer(6,'OutputMode','last')  % LSTM层                            
         reluLayer                         % Relu激活层                                
         fullyConnectedLayer(4)            % 全连接层 四个输出，值为4
         softmaxLayer                      % 分类层
         classificationLayer];

% 设置参数
options = trainingOptions('adam', ...            % Adam 梯度下降算法
          'MaxEpochs', 1200, ...                 % 最大训练次数 1200
          'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
          'LearnRateSchedule', 'piecewise', ...  % 学习率下降
          'LearnRateDropFactor', 0.1, ...        % 学习率下降因子为0.1
          'LearnRateDropPeriod', 950, ...        % 800次训练后 学习率为初始学习率*学习率下降因子
          'Shuffle', 'every-epoch', ...          % 打乱数据集
          'ValidationPatience', Inf, ...         % 关闭验证
          'Plots', 'training-progress', ...      % 画曲线
          'Verbose', false);

% 仿真预测
net = trainNetwork(pmm_train, tm_train, layers, options);
t_sim1 = predict(net, pmm_train);
t_sim2 = predict(net, pmm_test );
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

analyzeNetwork(net);

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
new_da =  double(reshape(new_da, 12, 1, 1, k1)); % 平铺数据
for i = 1:k1
    new_da_1{i,1} = new_da(:,:,1,i);
end
t_sim = predict(net,new_da_1); % 预测
tsim =vec2ind(t_sim');
xlswrite('预测结果',tsim'); % 得到预测值
