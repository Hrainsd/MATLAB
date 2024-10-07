function [value, wgt1, b1, wgt2, b2] = gacod(x)

%%  主空间变量
node1 = evalin('base', 'node1');             % 隐藏层神经元个数
net = evalin('base', 'net');                 % 网络参数
pm_train = evalin('base', 'pm_train');       % 输入数据
tm_train = evalin('base', 'tm_train');       % 输出数据

%%  初始化
Int_nd = size(pm_train, 1);                 % 输入节点数 
Out_nd = size(tm_train, 1);                 % 输出节点数

%%  权重编码
for i = 1 : node1
    for k = 1 : Int_nd
        wgt1(i, k) = x(Int_nd * (i - 1) + k);
    end
end

%%  输出权重编码
for i = 1 : Out_nd
    for k = 1 : node1
        wgt2(i, k) = x(node1 * (i - 1) + k + Int_nd * node1);
    end
end

%%  隐藏层偏置编码
for i = 1 : node1
    b1(i, 1) = x((Int_nd * node1 + node1 * Out_nd) + i);
end

%%  输出偏置编码
for i = 1 : Out_nd
    b2(i, 1) = x((Int_nd * node1 + node1 * Out_nd + node1) + i);
end

%%  赋值并计算
net.IW{1, 1} = wgt1;
net.LW{2, 1} = wgt2;
net.b{1}     = b1;
net.b{2}     = b2;

%%  模型训练
net.trainParam.showWindow = 0;      % 关闭训练窗口
net = train(net, pm_train, tm_train);

%%  仿真预测
t_sim1   = sim(net, pm_train);
tm_train = vec2ind(tm_train ); % 反归一化
tsim1    = vec2ind(t_sim1   );

%%  适应度值 
value = 1 ./ (1 - sum(tsim1 == tm_train) ./ size(pm_train, 2));
