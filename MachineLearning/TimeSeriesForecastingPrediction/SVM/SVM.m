%%SVM
%%
clc;clear
close all;
warning off;              

% 导入数据
result  = xlsread('时间序列预测数据.xlsx');
sps_num = length(result);  % 样本个数 

% 设置参数
sp = 50; % 步长
ts = 251; % 时间跨度(预测未来的个数)

%  构造数据
for i = 1:sps_num - sp - ts + 1
    res(i,:) = [reshape(result(i:i + sp - 1), 1, sp), result(i + sp + ts - 1)];
end

% 计算训练集和测试集的样本数量
train_percent = 0.8; % 训练集占比
test_percent = 1 - train_percent; % 测试集占比

train_samples = floor(train_percent * (sps_num - sp - ts + 1)); % 训练集样本数量
test_samples = sps_num - sp - ts + 1 - train_samples; % 测试集样本数量

% 计算训练集和测试集的索引
train_indices = 1:train_samples;
test_indices = train_samples + 1:sps_num - sp - ts + 1;

% 根据索引选择相应的样本
p_train = res(train_indices, 1:sp)'; % 训练集输入
t_train = res(train_indices, sp + 1)'; % 训练集输出

p_test = res(test_indices, 1:sp)'; % 测试集输入
t_test = res(test_indices, sp + 1)'; % 测试集输出

% 更新训练集和测试集样本数量
m = size(p_train, 2);
n = size(p_test, 2);

% 归一化处理
[pm_train,ps_input] = mapminmax(p_train,0,1); % 对训练集的自变量进行归一化
pm_test = mapminmax('apply',p_test,ps_input); % 'apply'、'ps_input'对测试集自变量进行归一化
[tm_train,ps_output] = mapminmax(t_train,0,1); % 对训练集的因变量进行归一化
tm_test = mapminmax('apply',t_test,ps_output); % 'apply'、'ps_input'对测试集因变量进行归一化

% 建立模型
pm_train = pm_train';pm_test = pm_test';
tm_train = tm_train';tm_test = tm_test';
c = 4.0; % 惩罚因子
g = 0.8; % 径向基函数参数
pst = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
pat = svmtrain(tm_train,pm_train,pst);
[t_sim1,error_1] = svmpredict(tm_train,pm_train,pat); % 预测
[t_sim2,error_2] = svmpredict(tm_test,pm_test,pat);
tsim1 = mapminmax('reverse',t_sim1,ps_output); % 'reverse'对输出的因变量进行反归一化
tsim2 = mapminmax('reverse',t_sim2,ps_output);

wrong1 = sqrt(sum((tsim1' - t_train).^2)./m); % 得到均方根误差
wrong2 = sqrt(sum((tsim2' - t_test).^2)./n); 

% 可视化
figure 
plot(1:m,t_train,'m-',1:m,tsim1,'c-','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
string = ['训练集预测结果对比：RMSE = ',num2str(wrong1)];title(string);
xlim([1,m]);legend('真实值','预测值');grid on;
saveas(gcf, '训练集预测结果对比.svg', 'svg'); % 保存为SVG文件

figure
plot(1:n,t_test,'m-',1:n,tsim2,'c-','LineWidth',1);
xlabel('预测样本');ylabel('预测结果');
string = ['测试集预测结果对比：RMSE = ',num2str(wrong2)];title(string);
xlim([1,n]);legend('真实值','预测值');grid on;
saveas(gcf, '测试集预测结果对比.svg', 'svg'); % 保存为SVG文件

size = 20;
color = 'c';
figure
scatter(t_train, tsim1, size, color);
hold on;
plot(xlim, ylim, '--k');
xlabel('训练集真实值');ylabel('训练集预测值');
xlim([min(t_train) max(t_train)]);ylim([min(tsim1) max(tsim1)]);
title('训练集真实值与预测值的对比图');
saveas(gcf, '训练集真实值与预测值的对比.svg', 'svg'); % 保存为SVG文件

figure
scatter(t_test, tsim2, size, color);
hold on;
plot(xlim, ylim, '--k');
xlabel('测试集真实值');ylabel('测试集预测值');
xlim([min(t_test) max(t_test)]);ylim([min(tsim2) max(tsim2)]);
title('测试集真实值与预测值的对比图');
saveas(gcf, '测试集真实值与预测值的对比.svg', 'svg'); % 保存为SVG文件

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
sps_num_new = length(new_da);  % 样本个数 

%  构造数据
for i = 1:sps_num_new - sp - ts + 1
    res_new(i,:) = [reshape(new_da(i:i + sp - 1), 1, sp), new_da(i + sp + ts - 1)];
end

new_data   = res_new(:,1:sp)';
new_da_1 = res_new(:,sp + 1)';
new_data   = mapminmax('apply',new_data,ps_input);
new_da_1 = mapminmax('apply',new_da_1,ps_output);
new_data   = new_data';
new_da_1 = new_da_1';
t_sim    =  svmpredict(new_da_1, new_data,pat);
tsim     =  mapminmax('reverse',t_sim,ps_output); % 'reverse'对输出的因变量进行反归一化
xlswrite('预测结果',tsim); % 得到预测值

% 绘制原始数据和预测结果的图形
figure;
hold on;
plot(1:sps_num_new, new_da, 'Color', '#0099CC', 'LineWidth', 1.5); % 绘制原始数据
plot(sps_num_new+1:sps_num_new+length(tsim), tsim, 'Color', '#99CCFF', 'LineStyle', '--' ,'Marker', '*'); % 绘制预测结果
hold off;
legend('原始数据', '预测结果');
xlabel('样本编号');
ylabel('数值');
title('原始数据和预测结果');
grid on;
saveas(gcf, '原始数据和预测结果.svg', 'svg'); % 保存为SVG文件
