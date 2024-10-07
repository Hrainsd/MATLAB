%%PSO-BP
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
tm_train = ind2vec(t_train);
tm_test  = ind2vec(t_test );

% 搭建网络
% 节点数
it_num  = size(pm_train, 1);  
hd_num = 6;                   
ot_num = size(tm_train, 1);   

net = newff(pm_train, tm_train, hd_num);
net.trainParam.epochs     = 1000;      % 训练次数
net.trainParam.goal       = 1e-6;      % 目标误差
net.trainParam.lr         = 0.01;      % 学习率
net.trainParam.showWindow = 0;         

% 初始化参数
l1     = 4.494;       % 学习因子
l2     = 4.494;       % 学习因子
maxge  =   30;        % 种群迭代次数  
sz_p   =    5;        % 种群规模
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
plot(1:length(fit_best),fit_best,'LineWidth',1);
xlabel('粒子群迭代次数');ylabel('适应度值');
xlim([1,length(fit_best)]);title('迭代误差变化');grid on;

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
