%%PSO-BP
%%
clear;clc;
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

% 搭建网络
it_num = size(pm_train,1);
hd_num = 5;
ot_num = size(tm_train,1);
net = newff(pm_train,tm_train,hd_num); 
net.trainParam.epochs = 1000;   % 迭代次数1000
net.trainParam.goal = 1e-6;     % 误差阈值1e-6    
net.trainParam.lr = 0.01;       % 学习率0.01
net.trainParam.showWindow = 0; 

% 初始化参数
l1     = 4.494;       % 学习因子
l2     = 4.494;       % 学习因子
maxge  =   60;        % 种群迭代次数  
sz_p   =    6;        % 种群规模
max_v  =  1.0;        % max—速度
min_v  = -1.0;        % min-速度
max_p  =  2.0;        % max-边界
min_p  = -2.0;        % min-边界
tt_num = it_num * hd_num + hd_num + hd_num * ot_num + ot_num;
for  i = 1 : sz_p     
    pp(i, :) = rands(1, tt_num);   
    v(i, :)  = rands(1, tt_num);   
    fs(i)    = fun(pp(i, :), hd_num, net, pm_train, tm_train);
end
[fs_zbest, idx_best] = min(fs);
zbest = pp(idx_best, :);     % 全局最佳
gbest = pp;                  % 个体最佳
fs_gbest = fs;               % 个体最佳适应度值
fit_best = fs_zbest;         % 全局最佳适应度值

% 迭代 
for i = 1 : maxge
    for j = 1 : sz_p
        % 更新速度
        v(j, :) = v(j, :) + l1 * rand * (gbest(j, :) - pp(j, :)) + l2 * rand * (zbest - pp(j, :));
        v(j, (v(j, :) > max_v)) = max_v;
        v(j, (v(j, :) < min_v)) = min_v;
        
        % 更新种群
        pp(j, :) = pp(j, :) + 0.2 * v(j, :);
        pp(j, (pp(j, :) > max_p)) = max_p;
        pp(j, (pp(j, :) < min_p)) = min_p;
        
        % 自适应变异
        pos = unidrnd(tt_num);
        if rand > 0.95
            pp(j, pos) = rands(1, 1);
        end
        
        % 适应度值
        fs(j) = fun(pp(j, :), hd_num, net, pm_train, tm_train);
    end
    
    for j = 1 : sz_p
        % 更新最优个体
        if fs(j) < fs_gbest(j)
            gbest(j, :) = pp(j, :);
            fs_gbest(j) = fs(j);
        end

        % 更新最优群体 
        if fs(j) < fs_zbest
            zbest = pp(j, :);
            fs_zbest = fs(j);
        end
    end
    fit_best = [fit_best, fs_zbest];    
end

% 初始最优化
w1 = zbest(1 : it_num*hd_num);
b1 = zbest(it_num*hd_num + 1 : it_num*hd_num + hd_num);
w2 = zbest(it_num*hd_num + hd_num + 1 : it_num*hd_num + hd_num + hd_num*ot_num);
b2 = zbest(it_num*hd_num + hd_num + hd_num*ot_num + 1 : it_num*hd_num + hd_num + hd_num*ot_num + ot_num);
net.Iw{1, 1} = reshape(w1, hd_num, it_num );
net.Lw{2, 1} = reshape(w2, ot_num, hd_num);
net.b{1}     = reshape(b1, hd_num, 1);
net.b{2}     = b2';

% 仿真预测
net.trainParam.showWindow = 1;       
net = train(net, pm_train, tm_train);
t_sim1 = sim(net, pm_train);
t_sim2 = sim(net, pm_test );
tsim1 = mapminmax('reverse',t_sim1,ps_output); % 'reverse'对输出的因变量进行反归一化
tsim2 = mapminmax('reverse',t_sim2,ps_output);

wrong1 = sqrt(sum((tsim1 - t_train).^2,2)'./m);   % 得到均方误差
wrong2 = sqrt(sum((tsim2 - t_test ).^2,2)'./n); 

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

figure
plot(1:length(fit_best),fit_best,'LineWidth',1);
xlabel('粒子群迭代次数');ylabel('适应度值');
xlim([1,length(fit_best)]);title('迭代误差变化');grid on;
saveas(gcf, '适应度变化曲线.svg', 'svg'); % 保存为SVG文件

% 指标结果
% 平均相对误差MBE
mbe1 = sum(tsim1 - t_train,2)'./m;
mbe2 = sum(tsim2 - t_test,2 )'./n;
disp(['训练集数据的平均相对误差为：',num2str(mbe1)]);
disp(['测试集数据的平均相对误差为：',num2str(mbe2)]);

% 平均绝对误差MAE
mae1 = sum(abs(tsim1 - t_train),2)'./m;
mae2 = sum(abs(tsim2 - t_test ),2)'./n;
disp(['训练集数据的平均绝对误差为：',num2str(mae1)]);
disp(['测试集数据的平均绝对误差为：',num2str(mae2)]);

% 决定系数R2
R1 = 1-norm(t_train - tsim1)^2 / norm(t_train - mean(t_train))^2;
R2 = 1-norm(t_test  - tsim2)^2 / norm(t_test  - mean(t_test ))^2;
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

new_data = res_new(:,1:sp)';
new_data = mapminmax('apply',new_data,ps_input);
t_sim  = sim(net,new_data); % 预测
tsim   = mapminmax('reverse',t_sim,ps_output); % 'reverse'对输出的因变量进行反归一化
xlswrite('预测结果',tsim'); % 得到预测值

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
