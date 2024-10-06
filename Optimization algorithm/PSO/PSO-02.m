%% 粒子群优化算法
%%
clc;clear;

%% 初始化种群  
f = @(x,y) (20 +  x.^2 + y.^2 - 10*cos(2*pi.*x)  - 10*cos(2*pi.*y)) ;

x0 = -5.12:0.05:5.12;
y0 = x0;
[X,Y] = meshgrid(x0,y0);
Z =f(X,Y);

figure(1); 
mesh(X,Y,Z);  
colormap("cool");

N = 50;                        % 初始种群个数  
d = 2;                         % 可行解维数  
ger = 50;                      % 迭代次数       
limit = [-5.12, 5.12];         % 设置位置限制  
vlimit = [-10, 10];            % 设置速度限制  
w = 0.8;                       % 惯性权重  
c1 = 0.5;                      % 个体学习因子  
c2 = 0.8;                      % 社会学习因子   

x = limit(1) + (  limit( 2 ) -  limit( 1)  ) .* rand(N, d);%初始种群的位置  
v = rand(N, d);                  % 初始种群的速度  
x_best = x;                      % 个体的历史最佳位置  
Nx_best = zeros(1, d);           % 种群的历史最佳位置  
Fx_best = ones(N, 1)*inf;        % 个体的历史最佳适应度   
FNx_best = inf;                  % 种群的历史最佳适应度  

hold on 
% [X,Y] = meshgrid(x(:,1),x(:,2));
% Z = f( X,Y ) ;
scatter3( x(:,1),x(:,2) ,f( x(:,1),x(:,2) ),'r*' );

%% 群体更新  
iter = 1;
record=[];
% record = zeros(ger,1); % 记录适应度的最大值

while iter <= ger  
     fx = f( x(:,1),x(:,2) ) ; % 个体当前适应度     
     for i = 1:N        
        if  fx(i) < Fx_best(i) 
            Fx_best(i)  = fx(i);    % 更新个体历史最佳适应度  
            x_best(i,:) = x(i,:);   % 更新个体历史最佳位置 
        end   
     end  
    if   min(Fx_best) < FNx_best
        [FNx_best, nmin] = min(Fx_best); % 更新群体历史最佳适应度  
        Nx_best = x_best(nmin, :);       % 更新群体历史最佳位置  
    end  
    v = v * w + c1 * rand * (x_best - x) + c2 * rand * (repmat(Nx_best, N, 1) - x);% 速度更新  
    
    % 处理边界速度 
    v(v > vlimit(2)) = vlimit(2);  
    v(v < vlimit(1)) = vlimit(1);  
    % 更新位置 
    x = x + v; 
    % 处理边界位置 
    x(x > limit(2)) = limit(2);  
    x(x < limit(1)) = limit(1);  

    record(iter) = FNx_best; 
    figure(2);  
    subplot(1,2,1);
    mesh(X,Y,Z);
    colormap("cool");
    hold on;
    scatter3( x(:,1),x(:,2) ,f( x(:,1),x(:,2) ) ,'r*');
    title(['状态位置变化','-迭代次数：',num2str(iter)]);
    subplot(1,2,2);
    plot(record);
    title('最优适应度变化');  
    pause(0.01)  
    iter = iter + 1; 
end  

figure(3);
mesh(X,Y,Z);
colormap("cool");
hold on;
scatter3( x(:,1),x(:,2) ,f( x(:,1),x(:,2) ) ,'r*');
title('最终状态位置');
disp(['最优值：',num2str(FNx_best)]);  
disp(['变量取值：',num2str(Nx_best)]);
