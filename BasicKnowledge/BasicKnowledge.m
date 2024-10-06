clc;clear

%%
a = "acer相机";
disp(a);
disp('acer相机');
b = 14;
abs(10-20);
char(65);
num2str(b);
num = 123.235346;
c = num2str(num);
disp(num2str(num));
disp(c);
class(c);

%%
c = [1,2,3;4,3,5;5,6,7];
d = c'; %c的转置矩阵
e = inv(c); %inv求方阵的逆矩阵
c*e;
E = zeros(10,5,3);
E(:,:,1) = rand(10,5); %都开区间
E(:,:,2) = randi(5,10,5); %生成(0-5)的10*5随机整数矩阵
E(:,:,3) = randn(10,5); %生成标准正态分布的随机矩阵
a = normrnd(0,2,10,5); % 生成正态分布均值为0，标准差为2的10×5的随机矩阵
b = roundn(3.1445926,-3); % 任意位 位置四舍五入
c = roundn([123.34,1245.689],-1);

%%
A = cell(1,6);
A{2} = eye(3);
A{5} = magic(5);
B = A{5};
C = cell(3,4,2);
A = rand(2,3,4,5);
sz = size(A);

%%
book = struct('name',{{'111' '222'}},'price',[11 22]);
book.name;
book.name(1);
book.name{1};

%%
A = [1 2 3 4 5 6]
B = 1:9
B = 1:2:9
C = repmat(B,3,4)
D = ones(2,4)

%%
A = magic(4)
B = 260*ones(4,4)
x1 = A\B
x2 = pinv(A)*B %pinv求伪逆矩阵
A + B
A - B
A * B
A * pinv(B)
A .* B
A ./ B

%%
A = magic(5)
B = A(2,5)
C = A(4,:)
D = A(:,4)
[m,n] = find(A>20) %找出大于20的值的行列数
corrcoef([250	4.60 	18.07
275	17.20 	17.28
300	38.92 	19.6
325	56.38 	30.62
350	67.88 	39.1
])

%%
%%for end
%求1-5的平方和
sum = 0;
for i = 1:5
sum = sum + i^2;
end
sum

%求1-5的阶乘的和
sum = 0;
for i = 1:5
    p = 1;
    for j = 1:i
        p = p*j;
    end
    sum = sum + p;
end
sum

%九九乘法表
for i = 1:9
    for j = 1:9
        a(i,j) = i*j;
    end
end
a

%%
%%while end
%求1+2+3+...+10的和
sum = 0;
n = 1;
while n < 11
    sum = sum + n;
    n = n + 1;
end
sum

%1-100的奇数的和
sum = 0;
n = 1;
while n < 101
    sum = sum + n;
    n = n + 2;
end
sum

% if 分支结构
a = 90
if a < 60
    "不及格"
elseif a < 90
    "良好"
else 
    "优秀"
end

%switch case end 
 month=3;
    switch month
        case {3,4,5}
            season='spring'
        case {6,7,8}
            season='summer'
        case {9,10,11}
            season='autumn'
        otherwise
            season='winter'

    end
num = 8;
switch num
    case {9,10}
        cj = '优秀'
    case {6,7,8}
        cj = '良好'
    case {4,5}
        cj = '一般'
    case {2,3}
        cj = '较差'
    otherwise
        cj = '差'
end

%%
x = 0:0.1:2*pi;
y = sin(x);
figure;
plot(x,y);
title('y = sin(x)');
xlabel('x');
ylabel('sin(x)');
xlim([0 2*pi]);

%
x = linspace(-2*pi,2*pi);
y1 = sin(x);
y2 = cos(x);
p = plot(x,y1,x,y2);

%
r = 2;
xc = 4;
yc = 3;
theta = linspace(0,2*pi);
x = r*cos(theta) + xc;
y = r*sin(theta) + yc;
plot(x,y);
axis equal;

%
r = 2;
theta = linspace(0,2*pi);
x =  2*cos(theta);
y =  2*sin(theta);
plot(x,y);
axis equal;

%
x = 0:0.01:20;
y1 = 200*exp(-0.05*x).*sin(x);
y2 = 0.8*exp(-0.5*x).*sin(10*x);
figure;
[AX,H1,H2] = plotyy(x,y1,x,y2,'plot');
set(get(AX(1),'Ylabel'),'String','Slow Decay');
set(get(AX(2),'Ylabel'),'String','Fast Decay');
xlabel('time musec');
title('multiple decay rates');
set(H1,'LineStyle','--','color','c','Marker','o');
set(H2,'LineStyle',':','color','b','Marker','*');

%%
t = 0:pi/50:10*pi
plot3(sin(t),cos(t),t)
xlabel('sin(t)')
ylabel('cos(t)')
zlabel('t')
grid on % hold on 保留当前绘图，可以添加新的绘图；hold off 删除当前绘图
axis square %绘图正方体

[x,y,z] = peaks(30);
mesh(x,y,z)
grid %grid on 有网格线
[X,Y,Z] = peaks; 
surf(X,Y,Z)
grid off

%%
a = input('请输入字符串：','s');
class(a);
a = input('请输入数字：\n');
class(a);

%%
% 图形窗口的分割
x = linspace(0,2*pi,60);
subplot(2,2,1);
plot(x,sin(x)-1);
title('sin(x)-1'); axis([0,2*pi,-2,0]);
subplot(2,1,2);
plot(x,cos(x)+1);
title('cos(x)+1'); axis([0,6,0,1]);
subplot(4,4,3);
plot(x,tan(x));
title('tan(x)'); axis([0,2*pi,-40,40]);
subplot(8,8,16);
plot(x,cot(x));
title('cot(x)'); axis([0,2*pi,-35,35]);

%%
%root 求根
% x^5-3x^3+2x^2-8x+33=0
a1 = [1 0 -3 2 -8 33];
x = roots(a1)

%多维运算
%2x+3y-z = 2
%8x+2y+3z = 4 
%45x+3y+9z = 23
a2 = [[2,3,-1];[8,2,3];[45,3,9]];
a3 = [2;4;23];
x = inv(a2)*a3

%求积分
f = @(x) x.^2
integral(f,0,1)

%%基本数据类型
%默认的数据类型就是double双精度浮点型 8个字节
a = 1+1i; %(1+1j)
aa = complex(6,7);
aaa = real(aa);
aaaa = imag(aa);
aaaaa = abs(aa);
sqrt(6^2+7^2);
angle(aa);%复数的辐角
conj(aa);%共轭运算（虚部取相反数）
format long; 
aa;

%矩阵
A = [1 2 3, 4, 5, 6+2i];
B = [7,8,9];
C = [A,B;B,A];
D = A*i;

%linspace
a = 1 + int64(9*rand(4,6,5));
a1 = linspace(0,2*pi,5);%linspace(1,2,3)1是开始，2是结束，3是平均分为多少个

%下标和序号
a(3,3);
a(7);
length(a) %6最大
ndims(a) %3维
numel(a) %4*6*5
a(:,3:2:6,2)

%reshape
f = [1 2 3 4 5 6 7 8]
f1 = reshape(f,4,2) %从列开始排

%显示变量信息who whos

%数学函数
x = sqrt(7)-2i;
y = exp(pi/2);
z = (5 + cosd(47))/(1+abs(x-y)); %cosd(角度)
log(10)
pow2(16) %幂运算
gcd(12,82) %最大公约数
factorial(5) %阶乘

%关系运算
% ==  % ~=等于和不等于
% & | ~与或非

%字符串
str1 = '0Aa'
str2 = '1Bb'
str1(2)
str3 = [str1;str2]
str3(1,3)
str3(4)
a = abs(str1)
char(a(1)+32)

%%
%结构矩阵
structA(1).ID = 001;
structA(1).name = 'Alen';
structA(1).data = [11,22;33,44];

structA(2).ID = 002;
structA(2).name = 'Alice';
structA(2).data = [11,22,55;33,22,66];

structA(3).ID = 003;
structA(3).name = 'Amy';
    name.small_name = 'liya';
    name.big_name = 'mi';
structA(3).data = [23,325;34,56];

structA(4).ID = 004;
structA(4).name = '486';
structA(4).data = [324,45;56,67];
structA(4).gender = 'male';
structA(4).char = 'courage' ;

structA

structB(1,1).ID = 010;
structB(1,1).name = 'wu';
structB(1,1).data = [32,34,55;65,57,34];

structB(1,2).ID = 014;
structB(1,2).name = 'fu';
structB(1,2).data = [34,426,43;325,346,124];

structB(2,1).ID = 016;
structB(2,1).name = 'nu';
structB(2,1).data = [1,45,56,56;57,56,59,56];

structB(2,2).ID = 034;
structB(2,2).name = 'ku';
structB(2,2).data = [324,46,346;56,46,435];
structB(2,2).gender = 'female';

structB

%索引
structA(3).name;
name.small_name;
structB(2,2).data(2,2);

%修改
structA(1).name = 'John';
name.big_name = 'York';
structA(4).data(1,1) = 36;

%删除
tempStruct = rmfield(structA,"char");
structA(4).char = [];

%%
%元胞cell
cellA = {1,'mike',[12,3,4;45,45,32];
         2,'kite',[21,23;34,35];
         3, 'alice',[34,64,12;56,342,234]};
cellA(2,3);
cellA{2,3};
structcellA.data1 = 'ruler';
structcellA.data2 = 'muder';
structcellA;
cellA{1,4} = 'MATLAB';
cellA{2,4} = structcellA;
cellA{3,4} = structA;

%显示整个元胞
celldisp(cellA)
cellplot(cellA)

%
cellA{4} = []; %值没了
cellA(4) = []; %矩阵断开了

%%
%特殊矩阵
zeros(3);
zeros(2,3);
exa = ones(3,4);
zeros(size(exa));
eye(3,4);
eye(5,3);
eye(3,3);
%建立[20,50]之间均匀分布的6阶随即矩阵
%y = a + (b-a) x
exa = 20 + (30*rand(6)); %int64可以整数化
%建立一个均值为0.6，方差为0.1的4阶正态随机分布矩阵
%y = μ +  σx
exa2 = 0.6 + sqrt(0.1)*rand(4);

%%
%特殊领域中的矩阵
%幻方矩阵 和为n(n^2+1)/2
magic(3);
%范德蒙德矩阵 任意子方阵可逆
V = [2,3,5,4,5];
vandermn = vander(V);
V1 = vandermn(3:5,1:3);
pinv(V1);
%希尔伯特矩阵 高度病态 （与阶数有关）
format rat;
h1 = hilb(4);
format;
h11 = invhilb(4);

%%
%diag为对角阵
diag(3)
diag(3,3); %第二个参数是在矩阵左下角添0
diag(1:3,1); 
A = [5,4,3,7,9];
DA = diag(A);
DA1 = diag(A,2);
B = eye(4);
B1 = B * 6;
C = magic(4);
V1 = diag(C);
V2 = diag(C,1); %取的是主对角线右上方的对角线上的元素

%建立一个5×5的矩阵，将其第一行元素×1，第二行×2...
A = magic(5);
B = diag(1:5);
C = B*A;
%建立一个4×4的矩阵，将其第一列元素×1，第二列元素×2...
A = magic(4);
B = diag(1:4);
C = A*B;
%三角阵
A3 = magic(7);
A3triup = triu(A3);
A3tril = tril(A3);
A3trilFra = triu(A3,5);

%转置与旋转
A = magic(5);
A';
%共轭转置
A = magic(4);
B = magic(4).*3-7;
C = A + B*i;
C';
%旋转rot90(A,K) 逆时针将A旋转K个90度
A = magic(4);
AR = rot90(A,1);
AR1 = rot90(A,4);
%左右翻转
F = magic(4);
F1 = fliplr(F);
%上下翻转
F = magic(4);
F1 = flipud(F);

%逆矩阵 a*a^-1 = I
%inv求方阵的逆 pinv求非方阵、奇异阵、非满秩矩阵
a = [1,2,3;2,5,7;3,6,4];
b = inv(a);
c = a*b;
a1 = [1,2,3,4;3,5,6,8]
b1 = pinv(a1)
c1 = a1*b1
%求解线性方程组x =a^-1 * b
%x+2y+3z=5;x+4y+9z=-2;x+8y+27z=6
a = [1,2,3;1,4,9;1,8,27];
b = [5;-2;6];
x = inv(a)*b;
rank(a);%求秩

%%
%矩阵的秩与迹（rank与trace）迹为方阵主对角线元素的和
a = magic(4);
a1 = rank(a);
ab = magic(3);
ab1 = trace(ab);
%向量和矩阵的范数
V = [1,2,3,-4];
norm(V,1); %所有元素绝对值之和
norm(V,2); %所有元素平方和的平方根
norm(V,inf); %所有元素绝对值中的最大值
V1 = [1,5,-8;3,5,1;3,7,9];
norm(V1,1); %所有列元素绝对值之和的最大值
norm(V1,2); %最大特征根的平方根
norm(V1,inf); %所有行元素绝对值之和的最大值
%矩阵的条件数
%衡量矩阵的病态程度，cond()求解 条件数=norm(A)*norm(inv(A)) 条件数一定大于1，越接近1越良性
A = rand(4);
A1 = hilb(4);
norm(A)*norm(inv(A)); %利用范数2来求解
norm(A1)*norm(inv(A1));
cond(A,1);
cond(A,2);
cond(A,inf);
cond(A1,1);
cond(A1,2);
cond(A1,inf);

%%
%矩阵特征值与特征向量
A = rand(5);
V = eig(A); %A的全部特征值
[X H] = eig(A); %X是特征值相对应的特征向量，H是特征值构成的对角阵

%%
%稀疏矩阵与稀疏存储方式
A = [2 0 0 0;
     0 0 0 0;
     0 -5 0 0;
     3 0 0 0 ];
%普通方式存储 2 0 0 3 0 0 -5 0 0 0 0 0 0 0 0 0
%稀疏方式存储(1,1),2,(4,1),3,(3,2),-5

%sparse() spconvert() spdiags()函数
s1 = sparse(A);
s2 = sparse(4,4); %4×4的全零稀疏矩阵
u = [1,3,4,4];%(1,3),1,(3,2),4,(4,1),5,(4,4),-8
v = [3,2,1,4];
s = [1,4,5,-8];
s3 = sparse(u,v,s);
full(s3);
[u1,v1,s0] = find(s3);
full(s3);

w = [2,3,1;  %(2,3),1,(3,5),1(2,4),2
     3,5,1;
     2,4,2];
s4 = spconvert(w);
full(s4);

location = [-1,0,1]; %主对角线以及左右两侧的位置
element = [1,3,5,7,0;
           2,4,6,8,10;
           0,1,1,1,1];
ele = element';
full(spdiags(ele,location,5,5));

%应用举例
%求解[2 3 0 0 0   [x1    [0
%    1 4 1 0 0     x2     3
%    0 1 6 4 0  *  x3  =  2
%    0 0 2 6 2     x4     1
%    0 0 0 1 1]    x5]    5]

loc = [-1,0,1]
el = [1,1,2,1,0;
      2,4,6,6,1;
      0,3,1,4,2]';
a = full(spdiags(el,loc,5,5));
b = [0;3;2;1;5];
x = inv(a)*b;

%%
%pause 暂停几秒
%选择结构if/switch
grade = input('请输入成绩：')
if grade >= 60 && grade <=80
    disp('成绩良好')
elseif grade <  60
    disp('成绩差')
else 
    disp("成绩优秀")
end

price = input('请输入价格：')
switch price
    case {1}
        disp('1')
    case {2}
        disp('2')
    otherwise
        disp('3')
end

%%
%循环结构
%for while break continue
%循环向量为空时，循环不执行；循环向量最终的值为向量最后的值
for k =[]
    disp('error')
end
for m =[1 2 3 9]
end
disp(m);
%按列对矩阵循环
%continue跳出一次循环，break跳出整层循环
for n =1:10
    if n == 5
        %break
        %continue
    else
        disp(n)
    end
end

%%
%递归
di_gui(5)
%测试可变参数
[a111,b111] = changepara(6,8)
%内联函数
ac = '(x^2-3*x+6*y+9*z)^3';
ac1 = inline(ac);
ac1(1,1,1);
%匿名函数
funcnone = @(x,y,z) x^2+y^2+z^2;
funcnone(2,2,2);
s = @sin;
s(pi/2);
disp('******************分界线******************')
%程序优化 利用探查器 Profiler
profile on;
genetic;
profile viewer;

%%
%二维绘图
x = 0:pi/100:2*pi;
y = 2*exp(-0.5*x).*sin(2*pi*x);
plot(x,y)

t = -pi:pi/100:pi;
x = t.*cos(3*t);
y = t.*sin(t).*sin(t);
plot(x,y)

x = -3*pi:pi/100:3*pi;%y的每个函数的横坐标都相同
y = [1000*sin(x);1000*cos(x);exp(x);10*2.^x];
plot(x,y)

t1 = -3*pi:pi/100:3*pi;
t2 = -pi:pi/100:pi;
x = [t1,t2;];%x的每列为横坐标，y的每列为纵坐标
y = [10*sin(t1),exp(t2)];
plot(x,y)

x = [1,2,3,4,5,10,9,8,7,-2];
plot(x)

x = 0:pi/100:2*pi;
y = exp(i*x);
plot(y)

x1 = -3:1/100:3;
x2 = -pi:pi/100:pi;
x3 = -4:1/100:4;
y1 = 5*cos(x1);
y2 = 5*sin(x2);
y3 = 5*sin(x3).*cos(x3);
plot(x1,y1,x2,y2,x3,y3)

%%
% 选项
%1.线型选项 -实线 -.点画线 :虚线 --双画线
%2.颜色选项 b蓝色 g绿色 r红色 c青色 m品红色 y黄色 k黑色 w白色
%3.标记符号选项 .点 o小圆圈 x叉号 +加号 *星号

x = 0:pi/100:2*pi;
y = 2*exp(-0.5*x).*sin(2*pi*x);
plot(x,y,'-.b*') %点画线蓝色星号

x1 = -3:1/100:3;
x2 = -pi:pi/100:pi;
x3 = -4:1/100:4;
y1 = 5*cos(x1);
y2 = 5*sin(x2);
y3 = 5*sin(x3).*cos(x3);
y4 = log10(x1);
y5 = 10*log10(x1);
figure
plot(x1,y1,':c+',x2,y2,'--mo',x3,y3,'-.k*')
grid on
figure 
plotyy(x1,y1,x3,y3) %plotyy绘制双纵坐标轴
grid on
figure
plot(x1,y4,'--m*',x1,y5,':c.')
grid on

x = 0:pi/100:2*pi;
y1 = 2*exp(-0.5*x).*sin(2*pi*x);
y2 = [2*exp(-0.5*x);-2*exp(-0.5*x)];
plot(x,y1,'r',x,y2,'--k');
grid on
title('信号波形及其包络'); %标题
xlabel('时间t'); %x轴名称
ylabel('信号幅值y'); %y轴名称
text(3.5,0.7,'可见包络随着震荡幅值减小而降低'); %图形说明
legend('信号波形','包络线'); %图例

x = 0:pi/100:2*pi
y1 = 2*exp(-0.5*x).*sin(2*pi*x);
y2 = [2*exp(-0.5*x);-2*exp(-0.5*x)];
plot(x,y1,'r',x,y2,'--k');
hold on %保持图形，叠加图层
plot(x,cos(x));

%%
%图形窗口的分割 subplot(m,n,location)
x = linspace(0,2*pi,60);
y1 = sin(x);
y2 = cos(x);
y3 = 2*exp(-0.5*x).*sin(2*pi*x);
y4 = sin(x).*cos(x);
subplot(2,1,2);
plot(x,y3,'m')
title('2*exp(-0.5*x).*sin(2*pi*x)');
grid;
subplot(2,2,1);
plot(x,y1,'b')
title('sin(x)');
grid;
subplot(4,4,3);
plot(x,y2,'c')
title('cos(x)');
grid;
subplot(4,4,7);
plot(x,y4)
title('sin(x).*cos(x)');
grid;
subplot(4,4,8);
plot(x,y3)
title('2*exp(-0.5*x).*sin(2*pi*x)');
grid;
subplot(8,8,8);
plot(x,y4,'k')
title('sin(x).*cos(x)');
grid;

%%
%特殊的二维绘图
%自适应采样(非等间隔采样)
x = 0:1/1000:1;
y = cos(tan(pi*x));
figure
plot(x,y);
figure
fplot(@(x) cos(tan(pi*x)), [0,1]); %自适应采样
%统计图 条形图 直方图 饼图  面积图 散点图 箱线图 色阶图
%条形图 bar()
bar([1 2 3 5 6]);
bar(magic(4)); %行数为组数，列数为一组有多少个对象值
A = [1 2 3;4 5 6;7 8 9;10 11 12];
subplot(1,2,1);
bar(A,'grouped');
title('Group');
subplot(1,2,2);
bar(A,'stacked');
title('Stack');

%直方图 hist()
y = randn(800,1);
subplot(2,2,1);
hist(y);title('高斯分布直方图'); %默认x为常量10

subplot(2,2,2);
x = 50
hist(y,x);title('指定常量的高斯分布直方图');

subplot(2,2,3);
x = -4:0.01:4;
hist(y,x);title('指定范围的高斯分布直方图');

theta = y*pi;
rose(theta);
title('极坐标系下的直方图');

%饼图 pie() 按照逆时针顺序分别为向量的各个元素
pie([3 5 7 9 20,16],[0 0 0 0 1 0]); %1表示分开
legend('优秀','良好','中等','一般','及格','差');

pie([3 9;5 20;7 16],[0 0 0 0 1 1]); %将矩阵按列的顺序排成一行，对行操作
legend('优秀','良好','中等','一般','及格','差');

%面积图 area()
x = 1:9;
y = [10 16 25 35 29 43 36 27 13];
area(x,y);
grid;title('面积统计图（向量）');

x = 1:2:9;
y = [1 3 5 7 9; 2 4 6 8 10;3 6 9 11 13]'; %第一列向量对x绘图 依次下一列与前面所有列的和对x绘图 
area(x,y);
grid;title('面积统计图（矩阵）');

%散点图 scatter() stairs() stem()
x = 0:0.1:3;
y = 2*exp(-0.5*x);
subplot(1,3,1);scatter(x,y,'m');title('scatter');
subplot(1,3,2);stairs(x,y,'c');title('stairs');
subplot(1,3,3);stem(x,y,'k');title('stem');

%箱线图 boxplot(x,g)
x = randi(100,10,2);
g = ['茶叶产量';'茶杯产量'];
boxplot(x,g);

%色阶图
%矩阵
matrix = [1, 2, 3, 4; 
          2, 1, 4, 5;
          3, 4, 1, 6;
          4, 5, 6, 1];
%创建自定义颜色映射向量（从天蓝到深蓝）
custom_map = [linspace(0.5, 0, 64);linspace(1, 0.5, 64);linspace(1, 0.8, 64)]';
colormap(custom_map); % 使用自定义颜色映射
%绘制色阶图
imagesc(matrix); % 绘制色阶图
colorbar; % 添加颜色刻度
%调整图形
xticks(1:4);
yticks(1:4);
xticklabels({'A', 'B', 'C', 'D'});
yticklabels({'A', 'B', 'C', 'D'});
xlabel('行');
ylabel('列', 'Rotation', 0);
title('矩阵色阶图');
%调整图像外边距
ax = gca;
ax.Position = [0.1, 0.1, 0.7, 0.7]; %调整图像位置和大小

%%
%三维图形的绘制 mesh 和 surf 就够了
%绘制x^2+y^2+z^2 = 64 ; y+z = 0这一空间曲线
t = 0:pi/50:2*pi;
x = 8*cos(t);
y = 4*sqrt(2)*sin(t);
z = -4*sqrt(2)*sin(t);
plot3(x,y,z);
title('plot3 函数举例');
xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

%
subplot(2,3,1);
x = 0:0.1:2*pi;
y = 0:0.1:2*pi;
[x,y] = meshgrid(x,y);% 生成平面网格坐标矩阵 x一行为一个副本 y一列为一个副本
z = sin(y).*cos(x);
mesh(x,y,z);
title('mesh函数绘制');xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

subplot(2,3,2);
x = 0:0.1:2*pi;
y = 0:0.1:2*pi;
[x,y] = meshgrid(x,y);% 生成平面网格坐标矩阵 x一行为一个副本 y一列为一个副本
z = sin(y).*cos(x);
surf(x,y,z);
title('surf函数绘制');xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

subplot(2,3,3);
x = 0:0.1:2*pi;
y = 0:0.1:2*pi;
[x,y] = meshgrid(x,y);% 生成平面网格坐标矩阵 x一行为一个副本 y一列为一个副本
z = sin(y).*cos(x);
plot3(x,y,z);
title('plot3函数绘制');xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

subplot(2,3,4);
x = 0:0.1:2*pi;
y = 0:0.1:2*pi;
[x,y] = meshgrid(x,y);% 生成平面网格坐标矩阵 x一行为一个副本 y一列为一个副本
z = sin(y).*cos(x);
meshc(x,y,z);
title('meshc函数绘制');xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

subplot(2,3,5);
x = 0:0.1:2*pi;
y = 0:0.1:2*pi;
[x,y] = meshgrid(x,y);% 生成平面网格坐标矩阵 x一行为一个副本 y一列为一个副本
z = sin(y).*cos(x);
meshz(x,y,z);
title('meshz函数绘制');xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

subplot(2,3,6);
x = 0:0.1:2*pi;
y = 0:0.1:2*pi;
[x,y] = meshgrid(x,y);% 生成平面网格坐标矩阵 x一行为一个副本 y一列为一个副本
z = sin(y).*cos(x);
surfl(x,y,z);
title('surfl函数绘制');xlabel('X');ylabel('Y');zlabel('Z',Rotation=0);
grid on

%
% 在xy平面内选择区域[-8,8]×[-8,8]，绘制函数z = sin(sqrt(x^2+y^2))/sqrt(x^2+y^2)
[x,y] = meshgrid(-8:0.5:8);
z = sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps); %eps是一个很小的数
subplot(2,2,1);
meshc(x,y,z);
title('meshc函数绘制');grid

[x,y] = meshgrid(-8:0.5:8);
z = sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
subplot(2,2,2);
meshz(x,y,z);
title('meshz函数绘制');grid

[x,y] = meshgrid(-8:0.5:8);
z = sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
subplot(2,2,3);
surfc(x,y,z);
title('surfc函数绘制');grid

[x,y] = meshgrid(-8:0.5:8);
z = sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
subplot(2,2,4);
surfl(x,y,z);
title('surfl函数绘制');grid

%
%球面绘制
subplot(2,2,1);
[x,y,z] = sphere(40); % 参数确定球面光滑程度，默认球体半径为1
surf(10*x,10*y,10*z); % 这时球体半径为10
title('球面');
%柱面绘制
subplot(2,2,2);
[x1,y1,z1] = cylinder(); % n为间隔点数量，默认为20
surf(x1,y1,z1);
title('柱面1');
subplot(2,2,3);
[x2,y2,z2] = cylinder(2+sin(0:pi/10:2*pi),40);
surf(x2,y2,z2);
title('柱面2');
%多峰面绘制
subplot(2,2,4);
[x3,y3,z3] = peaks(40);
surf(x3,y3,z3);
title('多峰面');

%
%隐函数绘图
subplot(2,2,1);
ezplot('x^2+y^3+6');grid;
subplot(2,2,2);
ezplot(@(x,y) x^2+y^3+6);grid;
subplot(2,2,3);
ezplot('cos(tan(pi*x))',[0,1]);grid;
subplot(2,2,4);
ezplot('8*cos(t)','4*sqrt(2)*sin(t)',[0,2*pi]);grid;
%案例x = e^(-s)*cos(t) y = e^(-s)*sin(t) z = t (0=<s<=8,0=<t<=5*pi)
ezsurf('exp(-s)*cos(t)','exp(-s)*sin(t)','t',[0,8,0,5*pi])

%图形修饰
%改变视角 view(az,el) az为方位角，el为仰角，默认az = -37.5度，el = 30度
subplot(1,2,1);mesh(peaks);
view(-37.5,30);
title('原视角');
subplot(1,2,2);mesh(peaks);
view(90,0);
title('侧视图');

%色图着色 cool spring summer autumn winter pink hot colorcube copper flag gray hsv jet bone lines prism white 
t = 0:pi/20:2*pi;z = peaks(50);
figure(1)
colormap('cool');surf(z);title('cool');
figure(2)
colormap('spring');mesh(z);title('spring');
figure(3)
colormap('summer');mesh(z);title('summer');
figure(4)
colormap('autumn');mesh(z);title('autumn');
figure(5)
colormap('winter');mesh(z);title('winter');
figure(6)
colormap('hot');surf(z);title('hot');
figure(7)
colormap('jet');surf(z);title('jet');

%平滑绘图 flat interp
t = 0:pi/20:2*pi;
z = peaks(40);
colormap('cool');
subplot(2,2,1);
mesh(z);
subplot(2,2,2);
surf(z);
subplot(2,2,3);
surf(z);shading flat;
subplot(2,2,4);
surf(z);shading interp

%裁剪
x = 0:pi/10:4*pi;
y = sin(x);
plot(x,y,'--k');
hold on ;
i = find(y > 0.5);
x(i) = NaN;
plot(x,y,'-b');

%%
%数据处理 第二个参数默认为1，为2时返回每行的处理结果(除去max和min)
% [max,index] = max() [min,index] = min() 
a = [1 2 3 5 9 10];
xmax = max(a);
xmin = min(a);
[a1,a11] = max(a) % a1为最大值，a11为其索引值
[a2 a11] = min(a) % a2为最大值，a22为其索引值

a = [1+2i 2+1i 9+6i 5+7i] % 复向量
[a1,a11] = max(a) % a1为复数模的最大值，a11为其索引值
[a2 a11] = min(a) % a2为复数模的最大值，a22为其索引值

a = magic(4);
max(a) % 以行向量的形式返回矩阵每列的最大值
max(max(a)) % 返回整个矩阵的最大值

% sum() 求和 第二个参数默认为1，为2时返回每行的和 prod() 求积 第二个参数默认为1，为2时返回每行的积
a = [1 2 3 4 5 6];
sum(a) 
a = [1 2 3;4 5 6];
sum(a) % 求矩阵每列的和
sum(a,2) % 求矩阵每行的和
sum(sum(a)) % 求矩阵所有元素的和
prod(a)
prod(a,2)
prod(prod(a))

%mean() 求平均值 median 求中值
a = [1 4 8;2 6 9;3 5 10];
mean(a)
mean(a,2)
mean(mean(a))
median(a)
median(a,2)
median(median(a))

%cumsum() 求累加 cumprod() 求累乘
a = [1 2 3;4 5 6;7 8 9];
cumsum(a)
cumsum(a,2)
cumsum(cumsum(a))
cumprod(a)
cumprod(a,2)
cumprod(cumprod(a))

% 标准差std() 方差var() 相关系数corrcoef() 协方差cov() 
% a = [1 2 3 4 5 6 7 8] b = [100 200 303 406 499 603 689 816] c = [-9 81 2 77 605 43 44 -88]

% 求矩阵每行的标准差
a = input('请输入矩阵1：');
[m,n] = size(a);
standard_ = 0;
x_ = mean(a,2);
for i = 1:m
    for j = 1:n
        standard_ = standard_ + (a(i,j) - x_(i))^2;
    end
end
Standard = sqrt(standard_/(n-1)); 
x = ['标准差为：',num2str(Standard)];
disp(x);
disp(std(a));
disp(var(a));

% 求矩阵每行的相关系数
a = input('请输入矩阵1：');
b = input('请输入矩阵2：');
[m,n] = size(a);
x_ = mean(a,2);
y_ = mean(b,2);
r_upper = 0;
r_lower1 = 0;
r_lower2 = 0;
for i = 1:m
    for j = 1:n
        r_upper = r_upper + (a(i,j) - x_(i))*(b(i,j) - y_(i));
        r_lower1 = r_lower1 + (a(i,j) - x_(i))^2;
        r_lower2 = r_lower2 + (b(i,j) - y_(i))^2;
    end
end
r = r_upper/sqrt(r_lower1*r_lower2);
x1 = ['相关系数为：',num2str(r)];
disp(x1);
disp(corrcoef(a,b)); % 主对角线为自相关系数 副对角线为相关系数

% 求矩阵每行的协方差
a = input('请输入矩阵1：');
b = input('请输入矩阵2：');
[m,n] = size(a);
x_ = mean(a,2);
y_ = mean(b,2);
c_upper = 0;
for i = 1:m
    for j = 1:n
        c_upper = c_upper + (a(i,j) - x_(i))*(b(i,j) - y_(i));
    end
end
cov_ = c_upper/(n-1);
x2 = ['协方差为：',num2str(cov_)];
disp(x2);
disp(cov(a,b)); %主对角线为自协方差 副对角线为协方差

% 排序 sort(x,dim,mode) 默认每列升序排序，即dim=1 mode=ascend
a= magic(4);
sort(a) 
sort(a,2,'descend') % 每行降序排列
[a_sorted,index] = sort(a,'descend') % 每列降序排序 index为索引值

%%
%多项式计算用向量的形式表示多项式 n次 用长度为 n+1 的行向量表示
%加减法
p1 = [1 2 1];
p2 = [1 -2 1];
p1 + p2  % 得到2*x^ + 0*x + 2*x^0
p1 - p2

%乘法 conv(p1,p2)
p1 = [1 8 0 0 -10];
p2 = [2 -1 3];
conv(p1,p2)

%除法 [Q,r] = deconv(p1.p2) Q返回商式 r返回余式
p1 = [1 8 0 0 -10];
p2 = [2 -1 3];
[q,r] = deconv(p1,p2)

%多项式求导 polyder()
% a = polyder(p)求p的导函数 
% a = polyder(p,q)求p*q的导函数 
% [a,b] = polyder(p,q)求p/q的导函数 a为导函数的分子 b为导函数的分母
p1 = 1;
p2 = [1 0 5];
[p,q] = polyder(p1,p2)

%多项式求值 polyval(p,x)
%polyval(p,x) p为多项式 x为自变量的值
p = [1 8 0 0 -10];
x = 1.2;
polyval(p,x)
x1 = [1.2 4 5 6;1.3 4.1 5.1 6.1];
polyval(p,x1) % 矩阵的每个元素表示每个x对应的多项式的值

%多项式求根 roots(p)
p =[1 2 1];
x = roots(p)

%%
%数据插值 
% 一维插值 interp1(x,y,x1,method) x：自变量 y：因变量 计算x1处的值 method：插值方法
% method：(默认为)linear(线性插值) nearest(最近点插值) pchip(hermite插值) spline(样条插值)
x = -2*pi:2*pi;
y = sin(x);
new_x = -2*pi:0.001:2*pi;
a = sin(new_x);
a1 = interp1(x,y,new_x);
a2 = interp1(x,y,new_x,"nearest");
a3 = interp1(x,y,new_x,"pchip");
a4 = interp1(x,y,new_x,"spline"); % 三次样条插值 画图最精准
a5 = interp1(x,y,2*pi:3*pi,'spline'); % 三次样条插值预测
subplot(1,5,1);
plot(x,y,'m',new_x,a1,'c');
subplot(1,5,2);
plot(x,y,'m',new_x,a2,'c');
subplot(1,5,3);
plot(x,y,'m',new_x,a3,'c');
subplot(1,5,4);
plot(x,y,'m',new_x,a4,'c');
subplot(1,5,5);
plot(x,y,'m',2*pi:3*pi,a5,'c');

% 二维插值 interp2(x,y,z,x1,y1,method)
x = [0:0.1:1];
y = [0:0.2:2];
[X,Y] = meshgrid(x,y);
z = X.^2 + Y.^2;
b1 = interp2(x,y,z,0.5,0.5)
b2 = interp2(x,y,z,0.5,0.5,"spline")

%%
% 曲线拟合所构造的拟合函数的次数小于插值结点的个数（matlab利用的是最小二乘原理）
%ployfit(x,y,n) % n为拟合函数的次数
x = linspace(0,2*pi,50);
y = sin(x);
p = polyfit(x,y,5)
p1 = polyfit(x,y,3)
x1 = 0:0.1:2*pi;
y1 = sin(x1);
y2 = -0.0056*x1.^5 + 0.0874*x1.^4 - 0.3946*x1.^3 + 0.2685*x1.^2 + 0.8797*x1 + 0.0102;
y3 = 0.0912*x1.^3 - 0.8596*x1.^2 + 1.8527*x1 - 0.1649;
plot(x1,y1,'r');grid on;hold on
plot(x1,y2,'g');grid on;hold on
plot(x1,y3,'b');grid on;
legend('sin(x)','5次拟合函数','3次拟合函数')

[x,y] = meshgrid([-6:0.1:6]);
z = x.^2 + y.^2;
colormap('jet');
surf(x,y,z);grid on
legend('椭圆抛物面')

%%
% 数值微分与数值差分
%数值微分diff(y,n) 计算x的n阶差分
x = linspace(0,2*pi,80);
y = sin(x);
chafen1 = diff(y)
chafen2 = diff(y,2)
chashang = chafen1/(2*pi/80)
plot(x,cos(x),'r');grid on;hold on;
plot(x,[chashang,1],'c');grid on;
legend('导数','数值导数');

%三种方法求导数值
%1 用高次多项式拟合函数，再对拟合函数求导，得到导数值
%2 直接求差商最为导数值
%3 先求导函数再带入点求值
f = @(x) sqrt(x.^3+2*x.^2-x+12)+(x+5).^(1/6)+5*x+2;
g = @(x) (3*x.^2+4*x-1)./sqrt(x.^3+2*x.^2-x+12)/2+1/6./(x+5).^(5/6)+5
x = -3:0.01:3;
p1 = polyfit(x,f(x),8);
p11 = polyder(p1);
p111 = polyval(p11,x)%第一种方法

p2 = diff(f([x,3.01]))/0.01%第二种方法

p3 = g(x)%第三种方法

plot(x,p111,'r',x,p2,'g',x,p3,'b');grid on;legend('多项式拟合求导','数值导数','解析导数');
 
%数值积分
%1 变步长simpson法
%[I,n] = quad(f,a,b,tol,trace) f为被积函数 a，b为积分下限和上限 tol为积分精度 trace为是否展现积分过程
%[I,n] = quadl(f,a,b,tol,trace)
% I为积分值 n为被积函数被调用的次数
f = @(x) exp(-x.^2);
[i,n] = quad(f,0,1)

%2 自适应积分法
% I = integral(f,a,b)
f = @(x) 1./(x.*sqrt(1-log(x).^2));
g = @(x) exp(-x.^2);
i = integral(f,0,1)
i = integral(g,0,1)

%多重定积分的数值求解
f = @(x,y) exp(-x.^2/2).*sin(x.^2+y);
i = integral2(f,-2,2,-1,1) % 内层是-2，2 外层是-1，1

%%
%线性方程组数值求解
%jacobi法(求解大型稀疏矩阵速度快)
A = [10 -1 0;-1 10 -2;0 -2 10];
b = [9 7 6]';
[x,n] = jacobi(A,b,[0 0 0]')

%直接求解法
A = [10 -1 0;-1 10 -2;0 -2 10];
b = [9 7 6]';
x = A\b

%非线性方程求解
%fzero函数 res = fzero(f,x0,tol,trace) f为待求函数 x0为搜索起点 
%求f(x) = x - 1/x + 5 在 x0 = -5 和x0 = 1 作为迭代初值
f = @(x) x-1./x+5;
res1 = fzero(f,-5)
res2 = fzero(f,1)
x = -6:0.1:6;
y = f(x);
plot(x,y,'b');grid on;title('f(x)');xlim([-6,1]);ylim([-6,6]);

%非线性方程组求解
%fsolve函数 res = fsolve(f,x0,option) option为优化参数 f为函数名 x0为初值
%求方程组在(1，1，1)附近的解并验证结果 sin(x)+y+z^2*exp(x) = 0;x+y+z = 0;x*y*z = 0
res = fsolve(@fsolve_f,[1,1,1])
test = fsolve_f(res)

%%
%最优化问题的求解
%[x,fminval] = fminbnd(f,x1,x2,option) % 求一元函数在区间[x1,x2]上的极小值点x和最小值fminval
%[x,fminval] = fminsearch(f,x0,option) % 求多元函数的极小值点和最小值，无导数算法，x0为向量
%[x,fminval] = fminunc(f,x0,option) % 求多元函数的极小值点和最小值，拟牛顿算法，x0为向量
%当目标函数阶数大于2，用后两个，当目标函数高度不连续时，用fminsearch
% -f 则求f的最大值
%求 f(x) = x-1/x+5在(-10,-1)和(1,10)的最小值点
f = @(x) x-1./x+5;
[x,fminval] = fminbnd(f,-10,-1)
[x,fminval] = fminbnd(f,1,10)
%求 f(x,y,z) = x+y^2/(4*x)+z^2/y+2/z在(0.5,0.5,0.5)附近的最小值
[x,fminval] = fminsearch(@fminsearch_,[0.5,0.5,0.5])

% Matlab 求解⾮线性规划的函数
[x,fval] = fmincon(@fun,XO,A,b,Aeq,beq,lb,ub,@nonlfun,option)
@nonlfun -- function [c,ceq] = nonlfun(x)
% c = [非线性不等式约束;...] ceq = [非线性等式约束;...]
% 注意把下标改写为括号 x1 === x(1)

%[x,fminval] = fmincon(f,x0,A,b,Aeq,beq,Lbnd,Ubnd,NonF,option) Lbnd为自变量的下限
%min f(x) = 0.4*x2+x1^2+x2^2-x1*x2+(1/30)*x1^3 
% s.t.{x1+0.5x2>=0.4;0.5x1+x2>=0.5;x1>=0,x2>=0}
f = @(x) 0.4*x(2)+x(1)^2+x(2)^2-x(1)*x(2)+(1/30)*x(1)^3;
x0 = [0;0];
A = [-1,-0.5;-0.5,-1];
b = [-0.4;-0.5];
Lbnd = [0;0];
[x,fminval] = fmincon(f,x0,A,b,[],[],Lbnd,[])

% 线性规划求解
[x,fval] = linprog(c,A,b,Aeq,beq,lb,ub,x0);
% 线性整数规划求解 注意把Aeq、beq、lb、ub写全
[x,fval] = intlinprog(c,intcon,A,b,Aeq,beq,lb,ub,x0);
% 最大最小化模型求解
[x,fval] = fminimax(@fun,x0,[],[],[],[],lb,ub,@nonlfun,option);

%linprog 
%[x,fbestval] = linprog(f,A,b,Aeq,Beq,Lbnd,Ubnd)
%x为最优解，fbestval为最优值,f为目标函数的系数向量
%min f(x) = -5x1-4x2-6x3
%s.t.{x1-x2+x3<=20;3*x1+2*x2+4*x3<=42;3*x1+2*x2<=30;x1>=0,x2>=0,x3>=0}
f = [-5,-4,-6]';
A = [1 -1 1;3 2 4;3 2 0];
b = [20;42;30];
Lbnd = [0;0;0];
[x,fbestval] = linprog(f,A,b,[],[],Lbnd)

%%
%常微分方程的数值解法 R-K方法
%[x,y] = 函数名(f,xspan,y0) ode为Ordinary differential equation
%函数名包括ode23、ode45、ode113、ode23t、ode15s、ode23s、ode23tb
%非刚性用：ode23和ode45 刚性用：ode15s、ode23s
%设有初值问题{y' = y^2-x-2/(4*(x+1)) 0=<x<=10;y0 = 2试求其数值解，并与精确解（y(x) = sqrt(x+1）+1）比较
x0 = 0;xf = 10;
y0 = 2;
[x,y] = ode45(@solver_f,[x0,xf],y0)
y1 = sqrt(x+1)+1;
plot(x,y,'c',x,y1,'r');
title('非刚性方程的数值解与解析解');grid on;legend('数值解','解析解');xlabel('X');ylabel('Y');

%%
%符号计算
%sym() syms x y z 空格隔开
a = sym('a');
s = (a+1)*2
a = sym(2);
s1 = a + 0.5
2+0.5

syms a b pi;
a = linspace(0,0.1,1);
b = 2*a^2+2;
y = sin(pi/3)+a*a*a+b*b;
% x = double(a);y = double(y);
% plot(x,y);


a = sym('a');
b = sym('b');
f = a*a*a+b*b ==120

syms x y;
s = 3*x*x*x-5*y*x+x*y*y+6 % 符号表达式
w = [1 x*x x*x*x;1 x*y y*y] % 符号矩阵

%符号运算 simplify()化简 factor()分解因式 expand()展开表达式  collect(s,x)按变量x合并同类项 
syms x y z
f = 2*x^2+3*x-5;
g = x^2-x+7;
f+g
h = (x^2-y^2)/(x-y);
simplify(h) % 化简

syms x y a b;
A = a^2-b^2;
factor(A)
factor(sym(166))
s = (x*x+y*y)*(x+y^2);
expand(s)
collect(s)
collect(s,y)
collect(s,x)

%符号运算中变量的确定
syms x y z a b;
h = 3;
s1 = 3*x+y+h
s2 = a*a*y+b
symvar(s1)
symvar(s1+s2)

%转置 对角阵 上三角 下三角
syms x;
A = [sin(x) cos(x);tan(x) cot(x)]
B = A'
C = diag(diag(A))
D = tril(A)
E = triu(A)

%%
%符号微积分
%求极限 
%limit(f,x,a)求函数f在自变量x趋近于a的极限
syms x m a;
f = (x^(1/m)-a^(1/m))/(x-a);
limit(f,x,a)
limit(f,x,a,'left')
limit(f,x,a,'right')
g = x^4+x^3+x^2+x;
limit(g,x,sqrt(2))

f = (sin(a+x) - sin(a-x))/x;
limit(f,x,0)

%求导数
%diff(f,x,n)求函数f对自变量x的n阶导
syms x y;
f = x^3+x^2+2*x;
diff(f,x,2)

f = x*cos(x);
diff(f,x,2)
diff(f,x,3)

%x = a*cos(t) y = a*sin(t) 求y'和y''
syms a t x y;
x = a*cos(t);
y = a*sin(t);
df1 = diff(y)/diff(x) %一阶导
df2 = (diff(y,2)*diff(x) - diff(y)*diff(x,2))/(diff(x))^2 %二阶导

%求积分
%int(f,x,a,b)函数f对x从a到b求积分
syms x y
f = 3*x^2;
int(f,x,1,3)

%%
%求解级数
%无穷级数求和 symsum(f,x,1,inf)表示求函数f自变量为x从1到无穷的和
syms n;
s1 = symsum(1/n^2,n,1,inf)
s2 = symsum((-1)^(n+1)*(1/n),n,1,inf)

%求解函数的泰勒级数
%taylor(f,x,0,'order',6)表示将目标函数f自变量为x在x=0处展开到前5阶（截断阶位6）的幂级数
syms x;
f = sqrt(1-2*x+x^3) - (1-3*x+x^2)^(1/3);
taylor(f,x,0,'order',4)

%方程符号求解
%代数方程求解
%[x,y,z] = solve([f1,f2,f3,f...],[x,y,z,...])
syms x;
f = 1/(x+2)+4*x/(x^2-4) == 1+2/(x-2); %求解方程1/(x+2)+4*x/(x^2-4)=1+2/(x-2)
x = solve(f,x)

syms x y;
f1 = x+y+3 == 2;f2 = x+2*y+1 == 6;
[x,y] = solve([f1,f2],[x,y])
syms x y; %求解方程组{1/x^3+1/y^3=28;1/x+1/y=4}
f1 = 1/x^3+1/y^3 ==28;
f2 = 1/x+1/y == 4;
[x,y] = solve([f1,f2],[x,y])

%常微分方程符号求解
%D表示导数，Dy表示y'，D3y表示y'''，Dy(0) = 5表示y'(0) = 5
%D3y+D2y+Dy-x+5 = 0 表示y'''+y''+y'-x+5=0
%dsolve(e,c,v) e是待求解ode c是初值 v是描述自变量的向量
y = dsolve('Dy-(x^2+y^2)/(2*x^2)','x') %求dy/dx = (x^2+y^2)/2*x^2的通解

y = dsolve('Dy-x^2/(1+y^2)','y(2)=1','x') %求dy/dx = x^2/(1+y^2)的特解，y(2)=1
 
who; % who可以显示所有变量
whos; % whos可以显示Name、Size、Bytes、Class、Attributes
a([1,3],:); %显示第1行和第三行的所有列的元素
a = [a,[1;2;3]]; %添加列向量
a = a(:); %得到一列向量，第一列，第二列放在第一列下面，以此类推...
a = floor(a); %向下取整
a = prod(a); %元素乘积
a = ceil(a); %向上取整
a = max(a,b); %取a,b中较大者，组成新的矩阵
a = max(a,[],2); %找出每行最大值，max(max(a))求矩阵所有元素最大值
a = flipud(a); %让矩阵上下翻转，第一行放在最后一行...
a = 0:0.01:5;b = sin(2*pi*a);c = cos(2*pi*a);plot(a,b,a,c); %画图

% 均值归一化
a = [1 2 3;4 5 6;7 8 9];
a_ = mean(a);
max_a = max(a);
min_a = min(a);
scal = max_a - min_a;
scald = (a-a_)./repmat(scal,3,1);

%for循环
a = zeros(10,2);
for j = 1:2
    for i = 1:10
        a(i,j) = i;
    end
end
disp(a);

%while循环
a = zeros(10,1);
i = 1;
while i < 6
    a(i) = a(i) + 6;
    i = i + 1;
end
disp(a);

%if elseif else语句
a = zeros(10,1);
a(3) = 6;
if a(3) == 6
    disp('值为6');
elseif a(3) == 0
    disp("值为0");
else
    disp('值未知');
end

%break、continue语句
a = zeros(10,1);
for i = 1:10
    a(i) = 2^i;
    if i == 5
        break
    end
end
disp(a);

a = zeros(10,1);
for i = 1:10
      if i == 5
        continue
      end
      a(i) = 2^i;
end
disp(a);
