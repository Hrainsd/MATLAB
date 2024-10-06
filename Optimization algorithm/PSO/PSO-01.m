%% 粒子群优化算法
%%
clc;clear;

%% 初始化种群 
f= @(x) (x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x)); % 函数表达式，求函数的最小值

N = 50;                         % 初始化种群个数
d = 1;                          % 可行解维数（自变量个数）
ger = 50;                       % 迭代次数
xlimit = [0, 50];               % 设置位置限制
vlimit = [-20, 20];             % 设置速度限制
w = 0.6;                        % 惯性权重
c1 = 0.5;                       % 个体学习因子
c2 = 0.8;                       % 社会学习因子

%绘制曲线
figure(1);
x0 = [0 ,xlimit(2)]; 
fplot(f,x0,color = '#4DD0E1');   

x = xlimit(1) + (  xlimit( 2 ) -  xlimit( 1)  ) .* rand(N, d); % 初始种群的位置
v = rand(N, d);                  % 初始种群的速度
x_best = x;                      % 个体的历史最佳位置
Nx_best = zeros(1, d);           % 种群的历史最佳位置  
Fx_best = ones(N, 1)*inf;        % 个体的历史最佳适应度  
FNx_best = inf;                  % 种群的历史最佳适应度  

hold on;
plot(x_best, f(x_best), 'ro');title('初始状态图');

%% 群体更新
iter = 1;
x1 = 0 : 0.01 : xlimit(2);
% record = zeros(ger, 1); % 记录适应度的最大值

while iter <= ger  
     fx = f(x) ; % 个体当前适应度     
     for i = 1:N        
        if fx(i) < Fx_best(i) 
            Fx_best(i) = fx(i);     % 更新个体历史最佳适应度  
            x_best(i,:) = x(i,:);   % 更新个体历史最佳位置 
        end   
     end  
    if  min(Fx_best)  < FNx_best 
        [FNx_best, n_best] = min(Fx_best);   % 更新群体历史最佳适应度  
        Nx_best = x_best(n_best, :);         % 更新群体历史最佳位置  
    end  
    v = v * w + c1 * rand * (x_best - x) + c2 * rand * (repmat(Nx_best, N, 1) - x); % 速度更新  

    % 处理边界速度  
    v(v > vlimit(2)) = vlimit(2);  
    v(v < vlimit(1)) = vlimit(1);  
    % 更新位置
    x = x + v;  
    % 处理边界位置  
    x(x > xlimit(2)) = xlimit(2);  
    x(x < xlimit(1)) = xlimit(1);  

    record(iter) = FNx_best;
    figure(2);
    subplot(1,2,1);
    plot(x1, f(x1), 'c-', x, f(x), 'ro');
    title('状态位置变化');
    subplot(1,2,2);
    plot(record);
    title('最优适应度变化') ; 
    pause(0.01);
    iter = iter + 1;  
end  
 
figure(3);
plot(x1, f(x1), 'c-', x, f(x), 'ro');title('最终状态位置');
disp(['最大值：',num2str(FNx_best)]);  
disp(['变量取值：',num2str(Nx_best)]);
