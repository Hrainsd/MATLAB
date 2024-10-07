%%GA-BP
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
pm_test = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
tm_train = ind2vec(t_train);
tm_test  = ind2vec(t_test );

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
tsim1 = vec2ind(t_sim1);
tsim2 = vec2ind(t_sim2);
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

figure
plot(tre(:, 1), 1 ./ tre(:, 2), 'LineWidth', 1.5);
xlabel('迭代次数');ylabel('适应度值');
title('适应度变化曲线');
grid on;

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
new_da = new_da';
new_da = mapminmax('apply',new_da,ps_input);
t_sim = sim(net,new_da); % 预测
tsim =vec2ind(t_sim);
xlswrite('预测结果',tsim'); % 得到预测值
