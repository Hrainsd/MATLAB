%%
% 第三讲 插值算法
yy = [133126,133770,134413,135069,135738,136427,137122,137866,138639, 139538];
xx = 2009:2018;
x2= 2019:2021;
x1 = 2009.5:2017.5;
y1 = interp1(xx,yy,x1,'spline');
y2 = interp1(xx,yy,x2,'spline');
subplot(1,2,1);
plot(xx,yy,'m',x1,y1,'-.c*');legend('原值','内部插值');xlabel('年份');ylabel('人口数据',Rotation=0);title('三次样条内部插值');grid on;
subplot(1,2,2);
plot(xx,yy,':o',x2,y2,'-*');legend('原值','外部插值');xlabel('年份');ylabel('人口数据',Rotation=0);title('三次样条外部插值');grid on;

%%
% 第四讲 拟合
x = [4.20000000000000
5.90000000000000
2.70000000000000
3.80000000000000
3.80000000000000
5.60000000000000
6.90000000000000
3.50000000000000
3.60000000000000
2.90000000000000
4.20000000000000
6.10000000000000
5.50000000000000
6.60000000000000
2.90000000000000
3.30000000000000
5.90000000000000
6
5.60000000000000];
y = [8.40000000000000
11.7000000000000
4.20000000000000
6.10000000000000
7.90000000000000
10.2000000000000
13.2000000000000
6.60000000000000
6
4.60000000000000
8.40000000000000
12
10.3000000000000
13.3000000000000
4.60000000000000
6.70000000000000
10.8000000000000
11.5000000000000
9.90000000000000];
polyfit(x,y,3);
f = @(x) 0.112414415921065*x.^3 + -1.674448037152997*x.^2 + 10.078635937108100*x - 13.108129717297286;
y11 = 0.112414415921065*x.^3 + -1.674448037152997*x.^2 + 10.078635937108100*x - 13.108129717297286;
fplot(f,[2,8],'c');grid on;xlabel('x的值');ylabel('y的值');title('拟合图');hold on;
scatter(x,y,'r*');grid on;legend('拟合函数','原函数');

% SST = SSE + SSR 总体平方和=误差平方和+回归平方和
SST = sum((y-mean(y)).^2);
SSE = sum((y-y11).^2);
SSR = sum(((y11-mean(y)).^2));

%%
% 第五讲 相关系数
% corrcoef()为皮尔逊相关系数 corr为斯皮尔曼相关系数
% 斯皮尔曼适用范围广 皮尔逊适用于连续、正态、线性关系的数据

% 用循环检验所有列的数据是否为正态分布
% 原假设为随机变量服从正态分布 h等于1,表示拒绝原假设 h等于0,表示不能拒绝原假设。
n_c = size(Test,2);
H = zeros(1,6);
P = zeros(1,6);
for i = 1:n_c
[h,p] = jbtest(Test(:,i),0.05);
H(i)=h;
P(i)=p;
end
disp(H)
disp(P)
[R,P] = corrcoef(Test)
[m,n] = size(Test);
a1 = mean(Test);
cov1 = zeros(m,n);
for j =1:n
    for i =1:m
        cov1(i,j) = Test(i,j) - a1(j);
    end
end
S = zeros(1,n);
for j=1:n
        S(1,j) = (sum(cov1(:,j).*cov1(:,j)))/(m-1);
end
cov2 = zeros(n,n);
r = zeros(n,n);
for i =1:n
    for j =i:n
        cov2(j,i) = (sum(cov1(:,i).*cov1(:,j)))/(m-1);
        r(j,i) = cov2(j,i)/(S(1,i)*S(1,j));
    end
end
for i =1:n
    for j =i:n
        r(i,j) = r(j,i);
    end
end
r % r为皮尔逊相关系数

[R1,P1]=corr(Test, 'type' , 'Spearman')

%%
% 第六讲 典型相关性分析 SPSS实现

%%
% 第七讲 多元回归分析 STATE实现

%%
% 第八讲 图论最短路径问题
% [P,d] = shortestpath(G,s,t,'Method',algorithm)
% 在图G中找到s节点到t节点的最短路径P和最短距离d
% dt = distances(G[,'Method',algorithm])返回任意两点的距离矩阵
% G = digraph([],[],weights[])有向图
% G = graph([],[],weights[])无向图
% highlight(p,P,'Edgecolor','b')表示高亮蓝色显示图p中路径P
A = [1 1 1 2 3 3 4 5 5 5 5 6 6 7 9 9];
B = [2 3 4 5 2 4 6 4 6 7 8 5 7 8 5 8];
W = [6 3 1 1 2 2 10 6 4 3 6 10 2 4 2 3];
G = digraph(A,B,W);
x = [-1 -0.5 -0.5 -0.5 0 0 0.25 0.5 0.5];
y = [0 0.5 0 -0.5 0.5 -0.5 -0.25 0 0.5];
dt = distances(G);
subplot(1,2,1);
plot(G);title('图论图');grid on;
subplot(1,2,2);
p = plot(G,'XData',x,'YData',y,'EdgeLabel',G.Edges.Weight,'EdgeColor','k');title('图论图');
[P,d] = shortestpath(G,1,8)
highlight(p,P,'EdgeColor','b');

%%
% 第九讲 分类模型 SPSS实现

%%
% 第十讲 聚类模型 SPSS实现 MATLAB实现
% 系统聚类(用于较多指标) DBSCAN聚类(用于两个指标)

%%
% 第十一讲 时间序列分析 SPSS实现
% 定义时间变量 （季节性分解） 时序图 时间序列建模器

%%
% 第十二讲 预测模型
% 灰色预测模型（数据是以年份度量的非负数据 数据能经过准指数规律的检验（除了前两期外，后面至少90%的期数的光滑比要低于0.5 数据的期数较短且和其他数据之间的关联性不强）
% 神经网络预测模型（大量数据）
%%
% 灰色预测GM(1,1)数据期数要大于3且小于10
% A = input('请输入需要预测的列矩阵：');
% A = [71.1 72.4 72.4 72.1 71.4 72.0 71.6]';
clc;clear;
X = [2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020]';
X1 = [2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023]'
A = [3693,3784,3841,3885,3945,3984,4016,4065,4104,4137,4161]';
[n,m] = size(A);
A_CUM = cumsum(A);
Z = (A_CUM(1:n-1) + A_CUM(2:n))/2;
B = zeros(n-1,2);
B(:,1) = -Z;B(:,2) = 1;
Y = A(2:n);
P = inv(B'*B)*B'*Y;
a = P(1); b =P(2);
num = input('请输入向后预测的期数：');
pre_1 = zeros(n+num,1);
for i = 1:n+num
    pre_1(i) = (A(1) - b/a)*exp(-a*(i-1)) + b/a;
end
pre_0 = zeros(n,1);
pre_0(1) = A(1);
for i = 1:n+num-1
    pre_0(i+1) = pre_1(i+1) - pre_1(i);
end
pre_1;
pre_0

% 级比检验
lmd = zeros(n-1,1);
for i =1:n-1
    lmd(i) = A(i)/A(i+1);
end
JDE_ = exp(-2/(n+1))<lmd<exp(2/(n+2));
JDE = double(JDE_);
if norm(JDE,1) == n-1
    disp('数据通过级比检验');
else
    disp('数据未通过级比检验，选择其他方法进行预测');
end

% 残差检验和级比偏差值检验
cancha = (A-pre_0(1:n))./A;
count = 0;
count_better = 0;
for i =1:n
    if cancha(i) < 0.1
        count_better = count_better + 1;
    if cancha(i) < 0.2
        count = count + 1;
    end
    end
    
end
if count == n && count_better == n
    disp('数据通过残差检验,且达到较高要求')
elseif count == n && count_better ~= n
    disp('数据通过残差检验,但未达到较高要求')
else
    disp('数据未通过残差检验，谨慎使用')
end

jibipiancha = 1-((1-0.5*a)/(1+0.5*a))*lmd;
count = 0;
count_better = 0;
for i =1:n-1
    if jibipiancha(i) < 0.1
        count_better = count_better + 1;
    if jibipiancha(i) < 0.2
        count = count + 1;
    end
    end
    
end
if count == n-1 && count_better == n-1
    disp('数据通过级比偏差值检验,且达到较高要求')
elseif count == n && count_better ~= n
    disp('数据通过级比偏差值检验,但未达到较高要求')
else
    disp('数据未通过级比偏差值检验，谨慎使用')
end

% 预测相对误差
err = abs((A-pre_0(1:n))./A);
ave_err = mean(err)
% 绘图
plot(X,A,'-b',X1,pre_0,'--c*');legend('原数据','预测值');title('灰色预测模型');xlabel('年份');ylabel('人口数（单位：千万人）');grid on;

%% 
% 第十三讲 奇异值分解SVD和图形处理
% 奇异值分解
A = input('请输入需要进行奇异值分解的原矩阵：');
[n,m] = size(A);% (n = m)
[U,S,V] = svd(A);
s = diag(S);
ro = input('请输入需要保留的特征比例（例如：0.9）：')
for i =1:n
    if sum(s(1:i))/sum(s) > ro
        disp(['特征比例为：',num2str(roundn(sum(s(1:i))/sum(s)))]);
        break
    end
end
A_ = U(:,1:i)*S(1:i,1:i)*V(:,1:i)' % 压缩后的矩阵

% 文件夹多张图片处理
fdr = 'C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\压缩文件夹内的所有图片\葫芦娃七兄弟';  
dt=dir(fullfile(folder_name, '*.jpg'));  
cell_name={dt.name};
n = length(cell_name);
ratio = 0.9;
for i = 1:n
    pto_name = cell_name(i);
    name = pto_name{1};  
    pto_add = fullfile(fdr, name); 
    save_add = fullfile(fdr,['compressed_',name]);
    pic = double(imread(pto_add)); % 灰色图片处理→pic = double(imread(pto_add));[real_ratio,compressed_pic] = oursvd(pic, ratio);imwrite(uint8(compressed_pic), save_add);
    R=pic(:,:,1); % red   
    G=pic(:,:,2); % green
    B=pic(:,:,3); % blue
    [real_ratio1,compressed_r] = oursvd(R, ratio);  
    [real_ratio2,compressed_g] = oursvd(G, ratio); 
    [real_ratio3,compressed_b] = oursvd(B, ratio); 
    compress_pic=cat(3,compressed_r,compressed_g,compressed_b);
    imwrite(uint8(compress_pic), save_add);
end

% 单张彩色图片SVD压缩
ratio = 0.9;
pho_add = "C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\千与千寻.jpg";
save_add = "C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\compressed_千与千寻.jpg";
pic = double(imread(pho_add));
R=pic(:,:,1); % red   
G=pic(:,:,2); % green
B=pic(:,:,3); % blue
[real_ratio1,compressed_r] = oursvd(R, ratio);  
[real_ratio2,compressed_g] = oursvd(G, ratio); 
[real_ratio3,compressed_b] = oursvd(B, ratio); 
compress_pic=cat(3,compressed_r,compressed_g,compressed_b);
imwrite(uint8(compress_pic), save_add);

% 单张灰色图片SVD压缩
ratio = 0.9;
pho_add = "C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\赫本.jpg";
save_add = 'C:\Users\23991\OneDrive\桌面\A01数学建模视频课程_解压版\第13讲.奇异值分解SVD和图形处理\代码和例题数据\奇异值分解SVD用于单张图片压缩\compressed_赫本.jpg';
pic = double(imread(pho_add));
[real_ratio,compressed_pic] = oursvd(pic, ratio);
imwrite(uint8(compressed_pic), save_add);

%%
% 第十四讲 主成分分析
clc;clear;
A = input('请输入矩阵（n个样本p个指标的n×p矩阵）：');
[n,p] = size(A);
B = input('请输入累计贡献率要求值（例如：0.8）：');
A1 = corrcoef(A);
[TZXL0,TZZ0] = eig(A1); % 特征向量，特征值
lmd0 = diag(TZZ0);[lmd,index] = sort(lmd0,'descend');TZXL = TZXL0(:,index);
ctn = zeros(p,1);
for i = 1:p
    ctn(i) = lmd(i)/sum(lmd);
end
cum_ctn = cumsum(lmd)/sum(lmd);
disp('贡献率为：');disp(ctn);
disp('累计贡献率为：');disp(cum_ctn);
disp('相对应的特征向量矩阵为：');disp(TZXL);
num = 1;
for i =1:p
    if cum_ctn(i) < B
        num = num + 1;
    else
        break
    end
end
disp(['只需要取前',num2str(num),'个主成分进行分析']);


%%
% 第十五讲 基于熵权法的topsis模型 在Topsis文件里

%%
% 第十六讲 因子分析 SPSS实现

%%
% 第十七讲 岭回归与lasso回归 STATE实现
% zscore()对数据进行标准化

%%
% 第十八讲 灰色关联分析 在GREY_correlation文件里

%%
% 第十九讲 弗洛伊德(Floyd)算法[stdt,path] 【最短距离，最短路径】
n = input('请输入节点数：');
W = ones(n)./zeros(n);
for i =1:n
    W(i,i) = 0;
end
W(1,2) = 3;W(1,3) = 8;W(1,5) = -4;W(2,4) = 1;W(2,5) = 7;W(3,2) = 4;W(4,1) = 2;W(4,3) = -5;W(5,4) = 6;
[stdt,path] = Floyd(W); 
% Floyd_onepath(path,stdt,3,1);
Floyd_path(W);

%%
% 第二十讲 ARCH和GARCH模型

%%
% 第二十一讲 正态分布均值假设检验
% 正态分布下z检验与t检验
A = input('请输入一行样本矩阵：');
alpha = input('请输入α的值：');
miu0 = input('请输入总体均值：');
n = size(A,2); % 1行n列矩阵
x_ = mean(A);
std_ = std(A);
B = input('总体方差是否已知（是或否）：','s');
if B == '是' % z检验
    sigma = input('请输入总体标准差：');
    z = (x_ -  miu0)*sqrt(n)/sigma;
    p = 1 - normcdf(z);
    if p < alpha
        disp('拒绝H0，即原假设不成立'); % H1是我们想要认为的
    else
        disp('不拒绝H0，即原假设成立');
    end
else % t检验
    t = (x_ - miu0)*sqrt(n)/std_;
    p = 1 - tcdf(t,n-1);
     if p < alpha
            disp('拒绝H0，即原假设不成立');
     else
            disp('不拒绝H0，即原假设成立');
     end
end
% 两个正态分布样本均值差的检验
A1 = input('请输入一行样本矩阵：');
A2 = input('请输入一行样本矩阵：');
alpha = input('请输入α的值：');
A = A1 - A2;
d_ = mean(A);
std_ = std(A);
t = d_*sqrt(n)/std_;
p = 1 - tcdf(t,n-1);
% if p < alpha
%     disp('拒绝H0，即原假设不成立');
% else
%     disp('不拒绝H0，即原假设成立');
% end

%% 第二十二讲 蒙特卡洛模拟
% f = @(x) x1*x2*x3的最大值
f = @(x) -x(1)*x(2)*x(3);
x0 = [20;10;-10];
A = [1 -2 -2;1 2 2;1 -1 0];
b = [0;72;10];
lbnd = [20;10;-10];
ubnd = [30;20;16];
[x,fminval] = fmincon(f,x0,A,b,[],[],lbnd,ubnd);
x
fmaxval = -fminval

% min f(x) =2*(x1^2)+x2^2-x1*x2-8*x1-3*x2
% s.t.
% (1) 3*x1+x2>9
% (2) x1+2*x2<16
% (3) x1>0 & x2>0
f = @(x) 2*x(1)^2 + x(2)^2 - x(1)*x(2) - 8*x(1) -3*x(2)
A = [-3 -1;1 2];
x0 = [3,3];
b = [-9;16];
lbnd = [0;0];
[x,fminval] = fmincon(f,x0,A,b,[],[],lbnd)

n=11111111;
x1=unifrnd(0,16,n,1);
x2=unifrnd(0,8,n,1);
fmin= +inf;
for i =1:n
    if 3*x1(i) + x2(i)>9 && x1(i) + 2*x2(i)<16
        fval =2*(x1(i)^2)+x2(i)^2-x1(i)*x2(i)-8*x1(i)-3*x2(i);
        if fval < fmin
            fmin = fval;
            num = [x1(i),x2(i)];
        end
    end
end
fmin % 起点处最小值
num % 得到起点
n=11111111;
x1=unifrnd(2,3,n,1);
x2=unifrnd(2,3,n,1);
fmin= +inf;
for i =1:n
    if 3*x1(i) + x2(i)>9 && x1(i) + 2*x2(i)<16
        fval =2*(x1(i)^2)+x2(i)^2-x1(i)*x2(i)-8*x1(i)-3*x2(i);
        if fval < fmin
            fmin = fval;
            num = [x1(i),x2(i)];
        end
    end
end
fmin % 得到蒙特卡洛模拟结果
num 

%%
% 第二十三讲 数学规划模型
% Matlab 求解⾮线性规划的函数
% [x,fval] = fmincon(@fun,XO,A,b,Aeq,beq,lb,ub,@nonlfun,option)
% @nonlfun -- function [c,ceq] = nonlfun(x)
% c = [非线性不等式约束;...] ceq = [非线性等式约束;...]
% 注意把下标改写为括号 x1 === x(1)

% 练习第一
A = [1,-1,1;3,2,4;3,2,0];
b = [20;42;30];
x0 = [0;0;0];
[x,fminval] = fmincon(@fun11,x0,A,b,[],[],[0;0;0])

% 练习第二
x0 = [0;0;0;0];
A =[-0.03,-0.3,0,-0.15;0.14,0,0,0.07];
b = [-32;42];
Aeq = [0.05,0,0.2,0.1];
beq = 24;
lb = [0;0;0;0];
[x,fminval] = fmincon(@fun22,x0,A,b,Aeq,beq,lb)

% 练习第三
x0 = [3;0;4];
A =[-2,5,-1;1,3,1];
b = [-10;12];
Aeq = [1,1,1];
beq = 7;
lb = [0;0;0];
[x,fminval] = fmincon(@fun33,x0,A,b,Aeq,beq,lb);
x
fminval = -fminval

% 练习1
A = [-2,3];b = [6];x0 = [1;1];
[x,fval] = fmincon(@fun,x0,A,b,[],[],[],[],@nonlfun);
x
fval_ = -fval

% 练习2
x0 = [2;4;0];
lb = [0;0;0];
[x,fval] = fmincon(@fun1,x0,[],[],[],[],lb,[],@nonlfun1)

% 练习3
x0 = [20,10,0]
aeq = [1,-1,0];beq = [10];A = [1,-2,-2;1,2,2];b = [0;72]
lb = [-inf;10;-inf];ub = [+inf;20;+inf];
[x,fval] = fmincon(@fun,x0,A,b,aeq,beq,lb,ub);
x
fval_ = -fval

% 线性规划求解
[x,fval] = linprog(c,A,b,Aeq,beq,lb,ub,x0);
% 线性整数规划求解 注意把Aeq、beq、lb、ub写全
[x,fval] = intlinprog(c,intcon,A,b,Aeq,beq,lb,ub,x0);
% 最大最小化模型求解
[x,fval] = fminimax(@fun,x0,[],[],[],[],lb,ub,@nonlfun,option);

% 例1
c=[-20,-10]';
intcon=[1,2]; % x1和x2限定为整数
A=[5,4;2,5];
b=[24;13];
lb=zeros(2,1); 
[x,fval]=intlinprog(c,intcon,A,b,[],[],lb);
x
fval = -fval

% 例2
c = [18;23;5];
intcon = 3;
A = [107,500,0;72,121,65;-107,-500,0;-72,-121,-65]
b = [50000;2250;-500;-2000]
lb = zeros(3,1);
[x,fval] = intlinprog(c,intcon,A,b,[],[],lb)

% 例3
c = [-3;-2;-1];
intcon = 3;
A = [1,1,1];
b = 7;
Aeq = [4,2,1];
beq = [12];
lb = zeros(3,1);
ub = [+inf;+inf;1];
[x,fval] = intlinprog(c,intcon,A,b,Aeq,beq,lb,ub)

% 例4 
c = [-540;-200;-180;-350;-60;-150;-280;-450;-320;-120];
intcon = [1,2,3,4,5,6,7,8,9,10];
A = [6,3,4,5,1,2,3,5,4,2];
b = 30;
Aeq = [];beq = [];
lb = zeros(10,1);ub = ones(10,1);
[x,fval] = intlinprog(c,intcon,A,b,Aeq,beq,lb,ub);
x
fval_ = -fval

% 例5
c = ones(7,1);
intcon = [1:7];
A = -[1 2 0 0 0 0 1;0 0 3 2 1 0 1;4 1 0 2 4 6 1];
b = [-100;-100;-100];
Aeq = [];beq = [];
lb = zeros(7,1);ub = [+inf;+inf;+inf;+inf;+inf;+inf;+inf];
[x,fval] = intlinprog(c,intcon,A,b,Aeq,beq,lb,ub)

% 例6
lb = [3;4];ub = [8;10];x0 = [3,4];
[x,fval] = fminimax(@fun3,x0,[],[],[],[],lb,ub);
x
fval = min(fval)

% 例7
c = [0.1466666;0.1566666];
A = [-1,-1];
b = -7;
lb = [0;0];ub = [5;6];
x0 = [4,4];
[x,fval] = linprog(c,A,b,[],[],lb,ub)

% 练习题1
c = ones(6,1);
intcon = [1:6];
A = -[1 0 0 0 0 1;1 1 0 0 0 0;0 1 1 0 0 0;0 0 1 1 0 0;0 0 0 1 1 0;0 0 0 0 1 1];
b = -[60;70;60;50;20;30];
lb = zeros(6,1);
[x,fval] = intlinprog(c,intcon,A,b,[],[],lb)

% 练习题2
c = -[3100;3100;3100;3800;3800;3800;3500;3500;3500;2850;2850;2850];
A = [1 1 1 0 0 0 0 0 0 0 0 0;0 0 0 1 1 1 0 0 0 0 0 0;0 0 0 0 0 0 1 1 1 0 0 0;0 0 0 0 0 0 0 0 0 1 1 1;1 0 0 1 0 0 1 0 0 1 0 0;0 1 0 0 1 0 0 1 0 0 1 0;0 0 1 0 0 1 0 0 1 0 0 1;480 0 0 650 0 0 580 0 0 390 0 0;0 480 0 0 650 0 0 580 0 0 390 0;0 0 480 0 0 650 0 0 580 0 0 390];
b = [18;15;23;12;10;16;8;6800;8700;5300];
lb = zeros(12,1);
[x,fval] = linprog(c,A,b,[],[],lb);
x
fval_ = -fval

% 练习题3
A = [-1 -2 0];b = [-1];
lb = [0;-inf;-inf];ub = [+inf;+inf;+inf];
[x,fval] = fmincon(@fun4,[0,0,0],A,b,[],[],lb,ub,@nonlfun4);
x
fval_ = -fval

% 练习题3
c = [1 1 1 1 1 1 ];
intcon = [1:6];
A = -[1 1 1 zeros(1,3);0 1 0 1 zeros(1,2);0 0 1 0 1 zeros(1,1);0 0 0 1 0 1;1 1 1 zeros(1,3);0 0 0 0 1 1;1 zeros(1,5);0 1 0 1 0 1];
b = -ones(8,1);
lb =  [zeros(1,6)];
ub = [ones(1,6)];
[x,fval] = intlinprog(c,intcon,A,b,[],[],lb,ub)
ss
