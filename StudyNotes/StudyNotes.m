%% MATLAB学习笔记
%% 常见数据类型及其操作

% 解释型语言是逐条编译的，编译型语言是一次编译的
clc;clear
x = -2 * pi: pi/360 : 2 * pi;
y = 2.^(-abs(x));
plot(x, y);

x = -2 * pi: pi/360 : 2 * pi;
y = sin(x);
plot(x, y);

x = -2 * pi: pi/360 : 1;
y = exp(x);
plot(x, y);

x = [2, 0, -3, 71, -9. 13];
y = roots(x);

A = [2, 3, -1;
    8, 2, 3;
    45, 3, 9];
b = [2; 4; 23];
x = inv(A) * b;

f = @(x) x;
integral(f, 0, 1);
f = @(x) x.*log(1 + x);
integral(f, 0, 1);

% complex,复数操作:real取实部、imag取虚部、abs取模、conj取共轭
a = 1 + 1i;
b = 1 + 1i;
c = a * b;
d = complex(6, 9);
e = real(d);
f = imag(d);
g = abs(d);
h = conj(d);

% class查看数据类型
class(d)

% 矩阵
% MATLAB是按列操作的
a = [1 2 3;
     4 5 6;
     7 8 9];
b = [2, 3, 5;
     2, 6, 1;
     4, 6, 8];
c = [a, b];
d = a + b*1i;

% 冒号表达式 start:walk:end
a = 1 + int64(9 * rand(4, 4));
b = 1:2:15;

% linspace(start, end, n), n表示在起始和终止之间生成多少个点, 间隔有n-1个, (end-start)/(n-1)就是步长
s = linspace(1, 10, 20);

% 索引, 按列索引
% 下标sub, 序号index
a = 1 + int64(9 * rand(4, 4, 5, 3));
a(2, 3);
a(3);
length(a) % 最大的维数
ndims(a) % 几维
numel(a) % 元素数目

% 矩阵中的:表示全部元素
a = 1 + int64(9 * rand(4, 4));
a(1, :);
a(1:2, 2:4);
a(1, 1:2:4);

% reshape
a = 1 + int64(9 * rand(4, 4));
b = reshape(a, 2, 8);

% 计算表达式的值
x = sqrt(7) - 2i;
y = exp(pi / 2);
z = (5 + cosd(47))/(1 + abs(x - y));

% log以e为底, log10以10为底
a = log(exp(1));
b = log10(100);

% 幂power函数
pow2(3);
power(3, 2);

% 最大公约数gcd函数
gcd(12, 24);

% 阶乘factorial
factorial(5);

% .* ./ * /
a = 1 + 9 * rand(4, 4);
b = 1 + 9 * rand(4, 4);
a .* b
a ./ b
a * b
a / b

% 关系运算符
% == ~=
% & | ~

% 字符串
str = '机器宇宙';
str0 = str(2);
str1 = 'I''m Joey';
disp(str1);
% 字符串数组
str2 = [str;'宇宙机器'];
str3 = str2(1, 2);
str4 = str2(2, 2);
a = abs(str) % 字符转化为ASCII码
b = char(a) % ASCII码转化为字符

%% 结构体和元胞

structa(1).ID = 01;
structa(1).Name = 'Mike';
structa(1).Data = [1, 2, 3, 4];

structa(2).ID = 02;
structa(2).Name = 'Jack';
structa(2).Data = [4, 2, 3, 4];

structa(3).ID = 03;
structa(3).Name = 'Joey';
structa(3).Data = [4, 2, 3, 5];

structa
% 结构矩阵的操作
% 索引
structa(3).Data(3);
% 修改
structa(3).Data(3) = 12;
structa(3).Data(3);
% 删除
structb = rmfield(structa, 'Data');

% 元胞
cella = {1, 'Mike', [1, 2, 3, 4];
         2, 'Jack', [4, 2, 3, 4];
         3, 'Joey', [4, 2, 3, 5]}
% 元胞的操作
% 索引 注意大小括号的区别
cella(2, 3);
cella{2, 3};
% 向元胞中加入结构矩阵
cella{1, 4} = 'MATLAB is a great software';
cella{2, 4} = structa;
cella{3, 4} = structa(2);
cella;
% 显示整个元胞
celldisp(cella);
cellplot(cella);

% {}和()区别很大
cella{4} = []; % 替换'Mike'这个数据为[]
cella(4) = []; % 删除'Mike'这个数据，使得元胞变为一个行向量

%% 通用矩阵与常见特殊矩阵

% 通用特殊矩阵 zeros() ones() eye() rand()0-1随机数 randn()均值为1方差为0的符合正态分布的随机数
zeros(3); % 3×3的方阵
zeros(1, 3); % 1×3的行向量

ones(3);
ones(1, 3);

eye(4);
eye(3, 4);
eye(5, 3);

y = 30 + 30 * rand(5); % [30, 60]内均匀分布的5阶随机矩阵
y = 0.6 + sqrt(0.1) * randn(4); % y = μ + σx, 得到均值为μ， 标准差为σ的4阶正态分布矩阵

% 特殊矩阵 魔方矩阵magic() 范德蒙矩阵vander() 希尔伯特矩阵hilb() invhilb()
magic(3); % 行的和 = 列的和 = 对角线的和
magic(4);

a = [1, 2, 3, 4, 5];
b = vander(a); % 最后一列全为1 倒数第二列为设置的参数向量 其余列为倒数第二列与前一列的对应元素的乘积
c = b(3:5, 1:3);
inv(c); % 任意子方阵可逆

a = hilb(4); % hi,j = 1/(i + j -1)
b = invhilb(4); % 高度病态(对矩阵的任何值做微小变化，都会使得该矩阵行列式的值或逆矩阵产生巨大的变化) 病态程度与阶数有关

%% 矩阵变换

% 对角阵
a = [1 2 3 4 5];
b = diag(a);
c = diag(a, 1);

% 数量矩阵
a = eye(5);
b = 3 * eye(5);

% 去对角线元素
a = magic(4);
b = diag(a);
c = diag(a, 1);

% 建立一个 5x5 的矩阵，将矩阵的第一行元素乘1，第二行元素乘2，……，第五行元素乘5
a = magic(5);
b = diag(1:5);
c = b * a;

% 建立一个 4x4 的矩阵，将矩阵的第一列元素乘1，第二列元素乘2，……，第四列元素乘4
a = magic(4);
b = diag(1:4);
c = a * b;

% 三角阵
% 上三角
a = magic(7);
b = triu(a);
% 下三角
c = tril(a);
% 小三角
d = triu(a, 2);

% 矩阵的转置与旋转
a = magic(4);
b = a'; % 或.'

% 旋转 rot90(A, K) 将矩阵A逆时针旋转K个90度
a = magic(5);
b = rot90(a, 2);

% 矩阵的逆和伪逆
% 矩阵的逆: 对于一个方阵A, 如果存在一个与相同阶的B,
% 使得A*B = B*A = I, 则称B与A互为逆矩阵
a = magic(5);
b = inv(a);
c = a * b;

% 求解线性方程组 Ax = b
% x = A^-1 * b
% 求解下列方程组
% x+2y+3z=5
% x+4y+9z=-2
% x+8y+27z=6
A = [1 2 3;
     1 4 9;
     1 8 27];
b = [5; -2; 6];
x = inv(A) * b;

% 伪逆
% 对于非方阵、奇异阵（是方阵，但行列式为0）、非满秩矩阵 (rank(A) ~= n )
a = [2 4 5 1;5 6 3 4; 2 5 7 3];
r = rank(a);
p = pinv(a);
b = a * p;

%% 矩阵求值

% 方阵的行列式 det()函数
a = magic(4);
b = det(a);

% 秩: 非零子式的最高阶数 rank()函数
% 秩的另一种解释: 将矩阵按列分分解为个向量，组成一个向量组。
% 则该向量组的最大线性无关组中包含向量的个数称为矩阵的行秩。
% 同理可定义列秩，可证明列秩=行秩=秩

% 秩可用于判断
% - 是否可逆
% - 初等变换后矩阵的秩不变
% - 判断非齐次线性方程组解的情况

a = magic(5);
r = rank(a);

% 迹: 矩阵主对角线元素的和 trace()函数
a = magic(5);
b = trace(a);

% 向量和矩阵的范数
% 向量或矩阵的范数用来度量其在某种意义下的长度。有多种定义，定义不同值也不同。
% 向量的范数
a = [1 2 3 4 5];
n1 = norm(a, 1) % 向量-1范数 元素绝对值之和
n2 = norm(a, 2) % 向量-2范数 元素平方和的平方根
n3 = norm(a, inf) % 向量-∞范数 所有向量元素绝对值中的最大值

% 矩阵的范数
a = magic(4);
n1 = norm(a, 1);% 矩阵-1范数 所有矩阵列元素绝对值之和的最大值
n2 = norm(a, 2); % 矩阵-2范数 矩阵最大特征值的平方根
n3 = norm(a, inf); % 矩阵-∞范数 所有矩阵行元素绝对值之和的最大值

% 矩阵的条件数
% 在求解线性方程组Ax = b时，考虑摄动对解的影响程度，影响大小取决于A及其逆矩阵。
% 引入条件数，用来描述误差的敏感程度，反映A矩阵的病态性
% 定义矩阵A的条件数为A的范数与A的逆的范数的乘积。条件数大于1，越接近于1越好，即越稳定。
F = rand(3);
FF = hilb(3);

% 条件数 cond()函数
cond(F, 1); % -1范数下的条件数
cond(F, 2); % -2范数下的条件数
cond(F, inf); % -∞范数下的条件数

cond(FF, 1);
cond(FF, 2);
cond(FF, inf);

% 矩阵的特征值与特征向量
% 对于n阶方阵A，λ为其特征值，假设A^T = A^T，则A的特征向量ξ为A的特征向量
a = magic(5);
lambda = eig(a); % 求解得到矩阵的全部特征值
[X, H] = eig(a); %  H为矩阵的全部特征值构成的对角阵, X各列是相对应的特征向量

%% 稀疏矩阵与稀疏存储方式

% 稀疏矩阵就是含有非常多零元素的矩阵
a = [2 0 0 0;
     0 0 0 0;
     0 1 0 0;
     0 0 0 4];
% 普通方式存储 2 0 0 0 0 0 1 0 0 0 0 0 0 0 0 4
% 稀疏方式存储 (1, 1), 2, (3, 2), 1, (4, 4), 4

% 稀疏存储方式 sparse()函数
b = sparse(a);
c = sparse(4, 5); % sparse(m, n)产生4×5的全0稀疏矩阵
u = [1 2 3 4];
v = [1 2 3 4];
S = [1 1 1 1];
d = sparse(u,v,S); % sparse(u, v, S)产生u行v列的S元素 
e = full(d); % full()产生完全存储方式的矩阵

% find()函数返回矩阵中非零元素的下标和值
a = [2 0 0 0;
     0 0 0 0;
     0 1 0 0;
     0 0 0 4];
[u, v, S] = find(a);

% spconvert()函数
a = [1 1 1;
     2 2 1;
     3 3 1;
     4 4 1];
b = spconvert(a); % spconvert(a)函数产生(1, 1)为1，(2, 2)为1，(3, 3)为1，(4, 4)为1的矩阵
c = full(b);

% 对角稀疏矩阵(梳状稀疏矩阵) spdiags()函数
a = [1 2 3 4;
     1 3 5 7;
     2 4 5 8]';
d = [-2, 0, 1]';
result = spdiags(a, d, 4, 4); % a以列存放某对角线的值，d指哪一条对角线
result = full(result);

% 求下列三对角线性方程组的解。
% [2 3        [x1  [0
%  1 4 1       x2   3
%    1 6 4     x3 = 2
%      2 6 2   x4   1
%        1 1]  x5]  5]
a = [1 1 2 1 0;
     2 4 6 6 1;
     0 3 1 4 2]';
b = [-1, 0, 1]';
A = full(spdiags(a, b, 5, 5));
B = [0; 3; 2; 1; 5];
x = inv(A) * B;

%% m文件的创建与使用，三种程序结构

% MATLAB中m文件分为两种，分别是脚本文件和函数文件
% 脚本文件实际上是个命令的集合，可以认为他是命令行的改良版，方便我们去编写命令
% 函数文件则是声明了一个函数，是一个代码块，方便我们调用
% 二者区别
% 脚本没有输入参数，也没有返回参数，而函数则是和其他语言一样，有输出参数，也可带有返回参数
% 脚本可以直接运行，而函数则需要调用才能运行

% 在命令行窗口直接输入脚本的文件名即可运行整个脚本文件
V = func1(5, 10);

% 三种程序结构 
% 顺序结构
a = 1;
A = [1 2;
     3 4;
     5 6];
disp(A); % disp()函数输出
pause(3); % pause()函数暂停x秒
disp(a);
% 编写求解一元二次方程的函数  
% ax^2+bx+c=0
[x1, x2] = func2(1, -2, 1);

% 选择结构
% if-elif-else, 顺序判断
a = 4;
if a < 2
    disp('值小于2');
elif a > 5
    disp('值大于5');
else
    disp('值大于等于2小于等于5');
end

% switch, 并行判断
a = 4;
switch a
    case a < 2
        disp('值小于2');
    case a > 5
        disp('值大于5');
    otherwise
        disp('值大于等于2小于等于5');
end

% 循环结构
% for while
% berak continue

% 阶乘 for循环
% 注意：循环在一开始进入的时候，循环变量就已经确定，不会再随着
% 循环体内部的改变而变化。
% 注意：最终的值就是循环体向量中最后的值
% 注意：循环向量为空的时候，循环一次也不会执行
res = 1;
n = 5;
for i = 1:1:n
    res = res * i;
    n = 10;
end
disp(res);

% for循环的循环变量为矩阵时, 按列循环
a = [1 2 3 4; 5 4 3 2; 3 4 5 6];
for k = a
    k;
end

% 阶乘 while循环
a = 5;
n = 1;
res = 1;
while n < 6
    res = res * n;
    n = n + 1;
end
disp(res);

% break和continue
% break用于跳出整层循环, 大break
% continue则是跳出一次循环, 小break
a = 10;
for k = 1:1:a
    if k == 5
%         break
        continue
    else
        disp(k);
    end
end

%% 循环嵌套

% for循环的嵌套
% ex1
n = 0;
for k = 1:1:10
    n = n + k;
end
disp(n);

% ex2
n = 0;
for i = 1:1:10
    for j = 1:1:10
        n = n + j;
    end
end
disp(n);

% ex3
n = 0;
for i = 1:1:10
    for j = i:1:10
        n = n + j;
    end
    disp(n);
    n = 0;
end

% ex4
res = 0;
m = 5;
for i = 1:1:m
    n = 1;
    for j = 1:1:i
        n = n * j;
    end
    res = res + n;
end
disp(res);

%% 函数文件的基本结构及其应用

% 函数文件的基本结构
% 函数的文件名和函数名可以不一致
% [输出参数表] = 函数名[输入参数]   
% 形参、实参  
% 注意：参数出现的顺序、个数要与定义时的相同

% 递归解决阶乘问题
res = func3(3);

% matlab中，函数参数的个数具有可变性  
% matlab的两个预定义变量，nargin，nargout
[res1, res2] = func4(6, 6);

% 全局变量global 以及局部变量local
% gloabl 作用域为整个函数空间，所有函数都可以对他进行存取和修改
% global可以用来函数间的信息传递
% 不提倡使用 global

%% 三种特殊函数及程序的调试与优化

% 子函数
% 主函数 Primary function
% 子函数 Subfunction
% 子函数类似于其他编程语言中的函数嵌套中的内层函数
res = func5(6, 4, 9);

% 内联函数
% 以字符串形式存在的函数表达式可以通过 inline 函数转化成内联函数
a = 'x + y + z';
b = inline(a);
c = b(1, 2, 3);
disp(c);

% 匿名函数 一行函数
% 匿名函数名 = @（匿名函数输入参数）匿名函数表达式 ex1
% 匿名函数名 = @函数名 ex2
f = @(a, b) a^2 + b^2;
res = f(1, 2);

s = @sind;
res = s(90);

%% 二维绘图方法及实例

% 普通二维函数图形
x = -2 * pi: pi/200: 2 * pi;
y = exp(x) .* 2.^(-3 * x);
plot(x, y);

% 参数方程的二维曲线
t = -2 * pi: pi/200: 2 * pi;
x = t .* cos(6 * t);
y = t .* sin(6 * t);
plot(x, y);

% 同时绘制三幅图像
x = -3 * pi: pi/200: 3 * pi;
y = [1000 * sin(x); 1000 * cos(x); exp(x)];
plot(x, y);

% 同型矩阵绘图 一列x对应一列y
t1 = -2 * pi: pi/100: 2 * pi;
t2 = -2: 1/100: 2;
x = [t1; t2]';
y = [sin(2 * t1); exp(t2)]';
plot(x, y);

% 折线图
x = magic(6);
plot(x(:, 1));

% 复平面的图像
x = 0: pi/100: 2 * pi;
y = exp(1i * x);
plot(y);

% 多个参数的plot()函数
x1 = -2: 1/100: 2;
y1 = sin(2 * x1);
x2 = -pi: pi/200: pi;
y2 = cos(2 * x2);
x3 = -3: pi/100: 3;
y3 = sin(x3) .* cos(x3);
plot(x1, y1, x2, y2, x3, y3);

%% 图形绘制选项及一些辅助操作

% plot中的各种选项
% 1.线性选项
% -实线 -.点画线 :虚线 --双画线

% 2.颜色选项
% r 红色 g 绿色 b 蓝色 m 品红色 y 黄色 c 青色 w 白色 k 黑色

% 3.标记符号选项
% .点 o小圆圈 x叉叉 +加号 *星号

x = -2 * pi: pi/200: 2 * pi;
y = sin(2 * x) .* cos(2 * x);
plot(x, y, '-.b*');

% 绘制包络
x = -pi: pi/200: 2 * pi;
y = 2 * exp(-0.5 * x) .* sin(2 * x);
z = [2 * exp(-0.5 * x); -2 * exp(-0.5 * x)];
plot(x, y,'-r', x, z, '--b');

% 双纵坐标 plotyy()函数
x = -2 * pi: pi/200: 2 * pi;
y1 = exp(-0.5 * x) .* sin(x);
y2 = sin(2 * x) .* cos(2 * x);
plotyy(x, y1, x, y2);

% 绘制图形的辅助操作
x = -pi: pi/200: 2 * pi;
y = 2 * exp(-0.5 * x) .* sin(2 * x);
z = [2 * exp(-0.5 * x); -2 * exp(-0.5 * x)];
plot(x, y,'-r.', x, z, '--b');
title('包络图');
xlabel('横坐标');
ylabel('纵坐标');
legend('真实线', '包络线');
text(2.5, 2.0, '信号变化');

% 图形保持 hold on
x = -pi: pi/200: 2 * pi;
y = 2 * exp(-0.5 * x) .* sin(2 * x);
z = [2 * exp(-0.5 * x); -2 * exp(-0.5 * x)];
k = sin(2 * x) .* cos(2 * x);
plot(x, y, '-r.', x, z, '--b');
hold on;
plot(x, k, ':co');

% 图形窗口分割
x = linspace(-2 * pi, 2 * pi, 800);
y1 = sin(x);
y2 = cos(x);
y3 = 5 * exp(-0.5 * x) .* sin(2 * x);
y4 = sin(2 * x) .* cos(2 * x);

subplot(2, 2, 1)
plot(x, y1, 'r');
title('sin(x)');
grid on;

subplot(2, 2, 2);
plot(x, y2, 'g');
title('cos(x)');
grid on;

subplot(2, 2, 3);
plot(x, y3, 'b');
title('5 * exp(-0.5 * x) .* sin(2 * x)');
grid on;

subplot(2, 2, 4);
plot(x, y4, 'c');
title('sin(2 * x) .* cos(2 * x)');
grid on;

% 灵活图形窗口分割
x = linspace(-2 * pi, 2 * pi, 800);
y1 = sin(x);
y2 = cos(x);
y3 = 5 * exp(-0.5 * x) .* sin(2 * x);
y4 = sin(2 * x) .* cos(2 * x);
y5 = sin(x) .* cos(3 * x);
y6 = exp(x);

subplot(2, 2, 1);
plot(x, y1, '-.r');
title('sin(x)');
grid on;

subplot(4, 4, 3);
plot(x, y2, 'g.');
title('cos(x)');
grid on;

subplot(4, 4, 4);
plot(x, y3, 'b.')
title('5 * exp(-0.5 * x) .* sin(2 * x)');
grid on;

subplot(4, 4, 7);
plot(x, y4, 'm.');
title('sin(2 * x) .* cos(2 * x)');
grid on;

subplot(4, 4, 8);
plot(x, y5, 'c.');
title('sin(x) .* cos(3 * x)');
grid on;

subplot(2, 1, 2);
plot(x, y6, ':k*');
title('exp(x)');
grid on;

%% 自适应采样及极坐标系绘图

% 采用自适应采样（非等间隔采样）fplot()函数
subplot(2, 1, 1);
x = 0:1/1000:1;
y = cos(tan(pi * x));
plot(x, y);

subplot(2, 1, 2);
fplot(@(x)cos(tan(pi * x)), [0, 1]);

% 绘制极坐标系蝴蝶曲线
t = 0: pi/50: 20 * pi;
r1 = exp(cos(t)) - 2 * cos(4 * t) + sin(t/12).^5;
r2 = exp(cos(t - pi/2)) - 2 * cos(4 * (t - pi/2)) + sin((t - pi/2)/12).^5;
subplot(1, 2, 1);
polarplot(t, r1);

subplot(1, 2, 2);
polarplot(t, r2, 'r');

%% 统计类二维图形的绘制

% 条形图
a = [1 3 5 2 1 9];
bar(a);

a = magic(4);
bar(a);

a = magic(3);
subplot(1, 2, 1);
bar(a, 'grouped');
title('Group');

subplot(1, 2, 2);
bar(a, 'stacked');
title('Stack');

% 直方图
a = randn(800, 1);
subplot(2, 2, 1);
hist(a);

subplot(2, 2, 2);
hist(a, 50);

subplot(2, 2, 3);
hist(a, -4:0.1:4);

subplot(2, 2, 4);
hist(a, -4:0.01:4);

% 极坐标图
x = randn(800, 1);
theta = pi * x;
rose(theta);

% 面积类图
% 饼图
% 注意: x可以为向量或矩阵。x为向量时, 按照逆时针顺序分别为x的各元素; 
% 若为矩阵, 则按照列的顺序依次排列矩阵元素为一行向量, 若x元素和小于一,则饼图是不完整的圆。
pie([2 4 6 8 10], [1 0 1 0 0]); % 1表示将该扇区突出显示
title('饼图');
legend('优秀','良好','中等','一般','差');

% 面积统计图
% 若x为向量，y为矩阵，则矩阵y的第一列对向量x绘图，然后依次是下一列与前面所有列的值的和对向量x绘图。
x = 1:1:10;
y = magic(10);
y = y(1, :);
area(x, y);

x = 1:1:10;
y = magic(10);
y = y(1:2, :)';
area(x, y);

% 散点类图
% scatter()散点图
% stairs()阶梯图
% stem()杆图
x = -5: 0.1: 5;
y = x.^2 + 2 * x + 1;
subplot(3, 1, 1);
scatter(x, y, 'r');

subplot(3, 1, 2);
stairs(x, y, 'g');

subplot(3, 1, 3);
stem(x, y, 'b');

%% 三维曲线与三维曲面的绘制

% 绘制三维线图 plot3(x1,y1,z1,x2,y2,z2,......)
% 当 x y z 是相同长度时，则 x y z 对应元素构成一条三维曲线；
% 当 x y z 是矩阵时，则 x y z 对应元素构成三维曲线，曲线条数等于矩阵个数；
% 绘制空间曲线：x^2 + y^2 + z^2 = 64 and y + z = 0
% x = 8cost and y = 4\sqrt{2}sint 0<=t<=2pi, z = -4\sqrt{2}sint
t = 0: pi/800: 2 * pi;
x = 8 * cos(t);
y = 4 * sqrt(2) * sin(t);
z = -4 * sqrt(2) * sin(t);
plot3(x, y, z, '.');
title('三维曲线');
xlabel('x轴');
ylabel('y轴');
zlabel('z轴');
grid on;

% 三维曲面
% 1 生成平面网格坐标矩阵
% 利用 meshgrid 函数生成
% 2 绘制三维曲面函数
% mesh()函数绘制三维网格图; surf()函数绘制三维曲面图
% mesh(x,y,z); surf(x,y,z)
% z 矩阵是网格点上的高度矩阵; x,y 是网格坐标矩阵
x = 0: pi/50: 2 * pi;
[x, y] = meshgrid(x);
z = sin(y) .* cos(x);

subplot(2, 3, 1);
mesh(z);
title('mesh');
grid on;

subplot(2, 3, 2);
surf(z);
title('surf');
grid on;

subplot(2, 3, 3);
plot3(x, y, z);
title('plot3');
grid on;

subplot(2, 3, 4);
meshc(z);
title('meshc');
grid on;

subplot(2, 3, 5);
meshz(z);
title('meshz');
grid on;

subplot(2, 3, 6);
surfl(z);
title('surfl');
grid on;

% 在 xy 平面内选择区域[-8, 8]x[-8, 8]
% 绘制函数 z = sin(sqrt(x^2 + y^2))/(sqrt(x^2 + y^2))
[x, y] = meshgrid(-8:0.2:8);
z = sin(sqrt(x.^2 + y.^2)) ./ (sqrt(x.^2 + y.^2) + eps); % eps是一个趋近于0的正数

subplot(2, 2, 1);
meshc(z);
title('meshc');
grid on;

subplot(2, 2, 2);
meshz(z);
title('meshz');
grid on;

subplot(2, 2, 3);
surf(z);
title('surf');
grid on;

subplot(2, 2, 4);
surfl(z);
title('surfl');
grid on;

% 3 标准三维曲面
% 球面
subplot(2, 2, 1);
[x, y, z] = sphere(50);
surf(x, y, z);
title('sphere');
grid on;

subplot(2, 2, 2);
[x, y, z] = cylinder(50);
surf(x, y, z);
title('cylinder1');
grid on;

subplot(2, 2, 3);
[x, y, z] = cylinder(2 + sin(0:pi/10:2 * pi), 30);
surf(x, y, z);
title('cylinder2');
grid on;

subplot(2, 2, 4);
[x, y, z] = peaks(30);
surf(x, y, z);
title('peaks')
grid on;

% 隐函数绘图 ezplot()函数
subplot(2, 2, 1);
ezplot('x^2 + y^2 -9');
grid on;

subplot(2, 2, 2);
ezplot(@(x, y) x^3 + y^2 -2 * x);
grid on;

subplot(2, 2, 3);
ezplot('sin(x) * cos(x)',[-2 * pi, 2 * pi]);
grid on;

subplot(2, 2, 4);
ezplot('2 * cos(2 * t)', '3 * sin(3 * t)', [-pi, pi]);
grid on;

%% 图形修饰与图像剪裁

% 视点处理
% 从不同的视点观察物体 view(az,el) az为方位角，el为仰角，默认az= -37.5，el=30
subplot(2, 2, 1);
mesh(peaks);
view(-37.5, 30);
title('视点1');

subplot(2, 2, 2);
mesh(peaks);
view(0, 90);
title('视点2');

subplot(2, 2, 3);
mesh(peaks);
view(90, 0);
title('视点3');

subplot(2, 2, 4);
mesh(peaks);
view(-7, -30);
title('视点4');

% 色图着色 colormap()函数
% summer cool pink hot spring copper
t = 0: pi/100: 2 * pi;
z = peaks(50);
colormap('summer'); 
surf(z);

% shading 平滑操作
% shading flat 平滑着色，较平滑; shading interp 插值着色，非常平滑
t = 0: pi/100: 2 * pi;
z = peaks(50);
colormap('summer'); 

subplot(2, 1, 1);
surf(z);

subplot(2, 2, 3);
surf(z);
shading flat;

subplot(2, 2, 4);
surf(z);
shading interp;

% 图形的裁剪处理
% 利用NaN常数剔除设置不可使用的数据
% example: 裁剪正弦函数大于θ.5部分的幅值
x = 0: pi/400: 2 * pi;
y = sin(x);
subplot(1, 2, 1);
plot(x, y, '-r.');

subplot(1, 2, 2);
x(find(y > 0.5)) = nan;
plot(x, y, '--c.');

% 图像读入/读出
img = imread("C:\Users\23991\OneDrive\图片\Acer\Acer_Wallpaper_02_3840x2400.jpg");
image(img);

[img, map] = imread("C:\Users\23991\OneDrive\图片\Acer\Acer_Wallpaper_02_3840x2400.jpg");

%% 数据分析与处理

% max()函数 min()函数
% 对于实数 对每列求最大、最小值以及值所对应列的索引
a = magic(5);
[a_max, max_ind] = max(a);
[a_min, min_ind] = min(a);

% 对于复数 对每列求模的最大、最小值以及值所对应列的索引
a = magic(5) + 1i;
[a_max, max_ind] = max(a);
[a_min, min_ind] = min(a);

% example: 求矩阵的最大值
a = magic(5);
a_max1 = max(a);
a_max = max(a_max1);

% example: 求矩阵的最大值及其对应的索引
a = magic(5);
a = reshape(a, 1, []);
[a_max, max_ind] = max(a);

% sum()函数
% 对每列求和
a = magic(5);
a_sum1 = sum(a);
a_sum2 = sum(a, 2);
a_sum = sum(sum(a));

% prod()函数
% 对每列求积
a = magic(5);
a_prod1 = prod(a);
a_prod2 = prod(a, 2);
a_prod = prod(prod(a));

% mean()函数
% 对每列求均值
a = magic(5);
a_mean1 = mean(a);
a_mean2 = mean(a, 2);
a_mean = mean(mean(a));

% median()函数
% 对每列求中值
a = magic(5);
a_median1 = median(a);
a_median2 = median(a, 2);
a_median = median(reshape(a, 1, []));

% cumsum()函数
% 对每列求累加
a = magic(5);
a_cumsum1 = cumsum(a);
a_cumsum2 = cumsum(a, 2);

% cumprod()函数
% 对每列求累乘
a = magic(5);
a_cumprod1 = cumprod(a);
a_cumprod2 = cumprod(a, 2);

% std()函数-标准差 var()函数-方差
a = magic(5);
stand = std(a)
vari = var(a)

% corrcoef()函数-相关系数
% 主对角线分别为 V1 和 V2 的自相关系数 
% 副对角线分别表示 V1 和 V2 的相关系数以及 V2 和 V1 的相关系数，二者相等
v1 = [2 3 5 1 4; 2 3 5 6 1];
v2 = [1 4 6 3 2; 3 6 8 1 5];
corr = corrcoef(v1, v2);

% cov()函数-协方差
% 主对角线分别代表 A 的自协方差和 B 的自协方差
% 副对角线分别代表 A 对 B 的协方差和 B 对 A 的协方差，二者相等
v1 = [2 3 5 1 4; 2 3 5 6 1];
v2 = [1 4 6 3 2; 3 6 8 1 5];
cov1 = cov(v1, v2);

% 排序
% sort(X,dim,mode)函数 默认对列升序
% mode: ascend升序 descend降序
% dim: 1对列排序 2对行排序
a = magic(5);
a1 = sort(a);
a2 = sort(a, 2);
a3 = sort(a, 'descend');

%% 多项式计算

% 在MATLAB中，n次多项式用一个长度为n+1的行向量表示，缺少的幂次项系数为0

% 1.加减法运算
% 即系数向量的加减运算，对于次数不同的多项式要把对应的空缺的高次项系数补0
% example:求（3x^3 + 5x^2 - 9）-（7x + 10） 
P1 = [3 5 0 -9];
P2 = [0 0 7 10];
P0 = P1-P2;
P0;

% 2.多项式乘法运算
% conv(P1,P2)函数 产生N1 + N2 - 1个数
% example: 求(x^4 + 8x^3 - 10)*(2x^2 - x+3)

P1 = [1 8 0 0 -10];
P2 = [2 -1 3];
P0 = conv(P1, P2);
P0;

% 3.多项式除法运算
% 函数[Q,r] = deconv(P1,P2)
% Q为商式，r为余式
% example: 求 (x^2 + 2x + 1)/(x + 1)
P1 = [1 2 1];
P2 = [1 1];
[Q, r] = deconv(P1, P2);
Q;
r;

% 4.多项式求导
% polyder()函数
% 1 p=polyder(P) 求多项式P的导函数
% 2 p=polyder(P,Q) 求多项式P*Q的导函数
% 3 [p,q]=polyder(P,Q) 求多项式P/Q的导函数,导函数的分子存入p，分母存入q
% example: 求 (x^2 + 2x + 1)的导函数
p0 = [1 2 1];
p = polyder(p0);

% 5.多项式求值
% polyval()函数 和 polyvalm()函数 输入参数为多项式系数向量和自变量x
% 前者用于代数后者用于矩阵
% 代数多项式求值
% example:已知多项式 x^4 +8x^3 -10 分别取x=1.2和一个2*3的矩阵为自变量
p = [1 8 0 0 -10];
x1 = 1.2;
x2 = [1 1.1 1.2;
      2 2.1 2.2];
y1 = polyval(p, x1);
y2 = polyval(p, x2);

% 6.多项式求根
% n次多项式可能有n个根，当然这些根可能是实根，也可能含有若干对共轭复根
% roots()函数可以用来求解全部根
% x=roots(P)
% example:求解多项式x^4+8x^3-10
p = [1 8 0 0 -10];
x = roots(p);

%% 数据插值与曲线拟合

% 一维数据插值
% Y = interp1(X,Y,X1,method)
% 函数根据X,Y的值，来计算函数在X1处的值
% method可选的选项如下所示
% linear 线性插值，默认的，将数据点直接直线连接
% nearest 最近点插值，优先最近的数据点进行插值操作
% pchip Hermite插值 采用分段三次多项式以及一阶导数相等的方法插值，更加光滑
% spline 在子区间构造三次多项式且节点处二阶导均连续，节点更加光滑

% 概率积分的数据表如下所示，用不同的插值方法计算f(0.472)
% f(x)=\frac{2}{\sqrt{\pi}}\int_{0}^{x}e^{-t^{2}}dt
% |  x   | 0.46      | 0.47      | 0.48      | 0.49      |
% | f(x) | 0.4846555 | 0.4937542 | 0.5027498 | 0.5116683 |
x=0.46:0.01:0.49;
y=[0.4846555,0.4937542,0.5027498,0.5116683];
format long;
y1 = interp1(x, y, 0.472, "linear");
y2 = interp1(x, y, 0.472, "nearest");
y3 = interp1(x, y, 0.472, 'pchip');
y4 = interp1(x, y, 0.472, "spline");

% 某检测参数f随时间t的采样结果如下所示，用数据插值法计算t=2，12，22，32，42，52时的f值
% t	0	5	10	15	20	25	30
% f	3.1025	2.256	879.5	1835.9	2968.8	4136.2	5237.9
% t	35	40	45	50	55	60	65
% f	6152.7	6725.3	6848.3	6403.5	6824.7	7328.5	7857.6
t = 0:5:65;
f = [3.1025, 2.256, 879.5, 1835.9, 2968.8, 4136.2, 5237.9, 6152.7, 6725.3, 6848.3, 6403.5, 6824.7, 7328.5, 7857.6];
t1 = 2:10:52;
y1 = interp1(t, f, t1, "linear");
y2 = interp1(t, f, t1, "nearest");
y3 = interp1(t, f, t1, 'pchip');
y4 = interp1(t, f, t1, "spline");

% 二维数据插值
% Z1＝interp2(X,Y,Z,X1,Y1,method)
% X Y为采样点 Z为采样值 X1, Y1为插值点
% 注意二维插值不支持Hermite插值方法(pchip)
% example: z= x^2 + y^2 在[0,1]*[0,2]区域内进行插值
x = 0:0.1:1;
y = 0:0.2:2;
[x, y] = meshgrid(x, y);
z = x.^2 + y.^2;
z1 = interp2(x, y, z, 0.5, 0.5, "linear");
z2 = interp2(x, y, z, 0.5, 0.5, "spline");

% 曲线拟合
% 最终求解的拟合函数是一个多项式，所以实际上求出来是一个多项式的系数向量
% 先用polyfit()函数求拟合函数
% 再用polyval()函数求某点近似值
% P=polyfit(X,Y,n)
% P为最终拟合函数的系数向量，X Y为采样点及采样点函数值，m为拟合多项式次数
% example: 分别用一个3次和5次多项式在区间[0,2*pi]内逼近函数 sinx
x = 0: 0.1: 2 * pi;
y = sin(x);
p1 = polyfit(x, y, 5);
p2 = polyfit(x, y, 3);
y1 = polyval(p1, x);
y2 = polyval(p2, x);

plot(x, y, 'ro');
grid on;
hold on;
plot(x, y1, 'b*');
hold on;
plot(x, y2, 'c*');
legend('sin(x)','5次多项式','3次多项式');
title('曲线拟合');

%% 数值微分与数值积分

% 两种方式计算f(x)在给定点x的数值导数，第一种是利用多项式对f(x)进行逼近，用逼近函数的导数求解x处的导数
% 第二种是利用f(x)在x处的差商作为其导数，Matlab中不能直接计算数值导数，但可以计算前向差分，函数为diff
% D1=diff(X) 计算向量X的前向差分
% Dx = diff(X,n) 计算X的n阶前向差分
% example:设X由[0,2*pi]间均匀分布的100个点组成 求sinx的1阶差分
x = 0: pi/50: 2 * pi;
y = sin(x);
y1 = cos(x);
d1 = diff(y, 1);
d2 = diff(y, 2);
d3 = diff(y, 3);

d1y = d1/(2 * pi / 100);

plot(x, y1, 'ro');
grid on;
hold on;
plot(x, [d1y, 1], 'c*');
title('二阶差商与解析导数');
legend('二阶差商', '解析导数');

% 三种方法
% 1 用高次多项式拟合函数f，再对拟合函数求导，再对导函数求值得到导数值
% 2 直接求取假设点的数值导数（差商）
% 3 先求出f'（x），再代入点求解导数值
f = @(x) sqrt(x.^3+2*x.^2-x+12)+(x+5).^(1/6)+5.*x+2;
g = @(x) (3*x.^2+4*x-1)./sqrt(x.^3+2*x.^2-x+12)/2+1/6./(x+5).^(5/6)+5;
x = -3:0.01:3;

y1 = polyfit(x, f(x), 8);
y1 = polyder(y1);
y1 = polyval(y1, x);

y2 = diff(f([x, 3.01])) / 0.01;

y3 = g(x);

plot(x, y1, 'c*', x, y2, 'b*', x, y3, 'ro');
legend('多项式拟合求导', '数值导数', '解析导数');
title('三种求导方法的比较');

% 数值积分原理
% 求任意函数在某一区间上的定积分的时候如果难以使用解析的方法求出原函数
% 可以考虑在区间内找到一个与被积函数近似的多项式函数，对该多项式函数进行积分求解，从而代替原函数的积分值，这就是数值积分 
% 求解方法
% 一般将待求积分区间分为n个等长子区间，在子区间内进行多项式的逼近，其逼近效果比整个区间上统一进行多项式近似的效果更好
% 这样就形成了数值积分复合公式，计算数学中已经证明在实际工程中采用复合Simpson积分公式已经具有足够的精度

% 1.变步长Simpson法
% [I,n] = quad(f,a,b,tol,trace)
% [I,n] = quadl(f,a,b,tol,trace)
% f为被积函数名，a,b为下限和上限，tol为积分精度（默认为10^-6），trace为是否展现积分过程（0则不展现）
% I为积分值，n为被积函数调用次数
% 注意：积分限不能为无穷大
f = @(x) exp(-x.^2);
[I, n] = quad(f, 0, 1);

% 2.自适应积分法
% I = integral(f,a,b)
% 注意：积分限可以为无穷大
f = @(x) exp(-x.^2);
I = integral(f, 0, 1);

% 多重定积分的数值求解
f = @(x, y) exp(-0.5* x.^2) .* sin(x.^2 + y);
I = integral2(f, -2, 2, -1, 1);

%% 线性方程及非线性方程的数值解

% 线性方程组求解

% 除法知识点回顾
% /右除：a/b表示矩阵a乘以矩阵b的逆 A*B^-1
% \左除：a\b表示矩阵a的逆乘以b A^-1*B
% ./右除：a./b表示矩阵a中的每个元素除以矩阵b的对应的元素
% .\左除：a.\b表示矩阵b中的每个元素除以矩阵a的对应的元素

% 1.利用左除运算符直接求解
% 对于线性方程组 Ax=b，可以通过 x=A\b来求解
% 2x₁ + x₂ - 5x₃ + x₄ = 13
% x₁ - 5x₂ + 7x₄ = -9
% 2x₂ + x₃ - x₄ = 6
% x₁ + 6x₂ - x₃ - 4x₄ = 0
A = [2 1 -5 1; 1 -5 0 7; 0 2 1 -1; 1 6 -1 -4];
b = [13; -9; 6; 0];
x = A\b;

% 2.迭代解法
% 先给定一个解的初始值，然后按照一定的迭代算法进行逐步逼近，求出更为精确的近似解
% 迭代法是一种不断用变量的旧值递推新值的过程，迭代法非常适合求解大型稀疏矩阵方程组，在数值分析中主要包括四种迭代法
% 分别是 Jacobi迭代法，Gauss迭代法，超松弛迭代法，两步迭代法

% 用 Jacobi 迭代法求解下列线性方程组。设迭代初值为 0，迭代精度为 10⁻⁶
% 10x₁ - x₂ = 9
% -x₁ + 10x₂ - 2x₃ = 7
% -2x₂ + 10x₃ = 6
A = [10 -1 0; -1 10 -2; 0 -2 10];
b = [9; 7; 6];
x0 = [0; 0; 0];
eps = 1e-6;
x = func6(A,b, x0, eps);

x = A\b;

% 非线性方程数值求解
% 单变量非线性方程求解
% fzero函数 z = fzero(f,x0,tol,trace) 
% f为待求函数，x0为搜索起点。一个函数可以有很多根，但fzero函数只能给出离x0最近的那一个根
f = @(x) x - 1./x + 5;
z1 = fzero(f, -5);
z2 = fzero(f, 1);

x = -6:0.01:2;
y = f(x);
plot(x, y, 'ro');
grid on;
hold on;
scatter([z1, z2], [f(z1), f(z2)], "cyan", "*");

% 非线性方程组求解
% fsolve()函数
% X = fsolve(f,x0,option)
% X为返回的解，f为函数名，x0为初值，option为优化参数

% 求下列方程组在(1, 1, 1)附近的解并对结果进行验证
% sin x + y + z²eˣ = 0
% x + y + z = 0
% xyz = 0
x0 = [1 1 1];
x = fsolve(@func7, x0)
f_val = func7(x)

%% 常见最优化问题的求解方法

% 无约束最优化问题求解
% 本质就是求解某个函数的最小值以及最小值的点
% 共三种函数，分别如下：
% 1 [x,fval] = fminbnd(f,x1,x2,options); % 求一元函数在区间[x1,x2]上的极小值点x和最小值fval
% 2 [x,fval] = fminsearch(f,x0,options); % 求多元函数的极小值点x和最小值fval（无导数算法），x0为向量
% 3 [x,fval] = fminunc(f,x0,options); % 求多元函数的极小值点x和最小值fval（拟牛顿法），x0为向量 
% 注意：当目标函数阶数大于2时，用fminsearch和fminunc，当目标函数高度不连续时用fminsearch
% 求解最大值则是求取-f的最小值得到的便是f的最大值

% 求函数f(x) = x - 1/x + 5在区间 (-10, -1) 和 (1, 10) 上的最小值点
f = @(x) x - 1./x + 5;
x1 = fminbnd(f, -10, -1);
y1 = f(x1);
x2 = fminbnd(f, 1, 10);
y2 = f(x2);

% 设f(x, y, z) = x + y²/4x + z²/y + 2/z
% 求函数f在(0.5, 0.5, 0.5)附近的最小值
[x, fval] = fminsearch(@func8, [0.5, 0.5, 0.5]);

% 有约束最优化问题求解
% 本质上就是在一定的条件下去求解函数的最小值（比如限制自变量的选取范围）
% fmincon()函数用于求解各种约束下的最优化问题
% [x,fval] = fmincon(f,x0,A,b,Aeq,beq,Lbnd,Ubnd,NonF,option)

% 求解有约束最优化问题
% min f(x) = 0.4x₂ + x₁² + x₂² - x₁x₂ + (1/30)x₁³
% s.t.  x₁ + 0.5x₂ ≥ 0.4
%       0.5x₁ + x₂ ≥ 0.5
%       x₁ ≥ 0, x₂ ≥ 0
f = @(x) 0.4*x(2) + x(1)^2 + x(2)^2 - x(1)*x(2) + x(1)^3/30;
x0 = [0, 0];
A = [-1 -0.5; -0.5 -1];
b = [-0.4; -0.5];
lb = [0, 0];
[x, fval] = fmincon(f, x0, A, b, [], [], lb, []);

% 线性规划问题求解
% 使用linprog()函数
% [x, fval] = linprog(f, A, b, Aeq, beq, Lbnd, Ubnd)
% x为最优解，fval为最优值，x, b, beq, Lbnd, Ubnd是向量，A, Aeq是矩阵，f为目标函数的系数向量

% 求解线性规划问题。
% min f(x) = -5x₁ - 4x₂ - 6x₃
% s.t. x₁ - x₂ + x₃ ≤ 20
%      3x₁ + 2x₂ + 4x₃ ≤ 42
%      3x₁ + 2x₂ ≤ 30
%      x₁ ≥ 0, x₂ ≥ 0, x₃ ≥ 0
f = [-5 -4 -6];
A = [1 -1 1; 3 2 4; 3 2 0];
b = [20; 42; 30];
lb = [0, 0, 0];
[x, fval] = linprog(f, A, b, [], [], lb, []);

%% 常微分方程的数值解与R-K数值解算法

% 未知函数是一元函数的微分方程称作常微分方程
% 未知函数是多元函数的微分方程称作偏微分方程
% 微分方程中出现的未知函数最高阶导数的阶数，称为微分方程的阶
% solver为某函数
% [t,y] = solver(f,tspan,y0) solver为求解函数
% 有七种常见的，ode23、ode45、ode113、ode23t、ode15s、ode23s、ode23tb
% t, y为对应的数值解，f为定义好的函数f(t,y)，该函数必须返回的是列向量，tspan为求解区间[t0,tf]，y0是初始状态向量

% 设有初值问题：
% y' = (y² - t - 2) / 4(t + 1),  0 ≤ t ≤ 10
% y(0) = 2
% 试求其数值解，并与精确解相比较（精确解为 y(t) = √(t+1) + 1）
f = @(t, y) (y^2 - t - 2) / (4*(t+1));
y0 = 2;
tspan = [0, 10];
[x, y] = ode45(f, tspan, y0);
t = 0:0.01:10;
g = sqrt(t + 1) + 1;
plot(x, y, '--c*');
grid on;
hold on;
plot(t, g, '-ro');
legend('数值解','解析解');
title('非刚性方程的数值解与解析解');

%% 符号计算

% sym建立单个符号对象
% 符号计算，即在运算时无需事先对变量赋值，而是将得到的结果用符号表示，可以得到比数值计算更一般的结果
a = sym(2);
b = a + 0.5;
c = sym(pi/3);
d = sin(c);

a = sym('a');
b = sym('b');
c = 5;
d = 2;
w = (a + b) * (a -b);
w = c * d;

% syms建立多个符号对象
syms x y a b pi; % 一次建立多个符号变量，注意：不需要加引号，要用空格隔开
z = sin(pi/3) + a * a + b * b + a * b;
f = 2 * a + 3 * b == 100; % 建立符号方程
z = 3 * x + 2 *y + x * y; % 建立符号表达式
w = [x y; x+y x-y; x*y x/y]; % 建立符号矩阵

%符号表达式的四则运算  
%用加减乘除幂次等运算符实现，运算结果依然是一个符号表达式
syms x y z
f = x^2 + 2 * y +3 * x;
g = 2 * x^2 + 3 * y + 2 * x;
f + g;

f = (x^2 - y^2)/(x + y); % 注意，matlab一般情况下并不会帮你化简

% 符号表达式的因式分解与展开
% matlab中提供了符号表达式的因式分解与展开的函数。调用格式如下
% factor(s);对符号表达式s分解因式
% expand(s);对符号表达式s进行展开
% collect(s);对符号表达式s合并同类项
% collect(s,v);对符号表达式s按变量v合并同类项
syms a b x y
s = a^2 - b^2;
factor(s);
f = (2 * x^2 + 3 * y^2) * (3 * x^2 - y^2);
expand(f);
g = x^3 + (2 * y - 5 * x) * x^2 + 4 * y^3;
collect(g);
collect(g, y);

% 符号运算中变量的确定
syms a b x y
c = 3;
s1 = a * x + b * y;
s2 = x^2 + c;
symvar(s1);
symvar(s2);
symvar(s1 + s2);

% 符号矩阵的化简
% simplify()函数
syms a b x y alpha;
m = [a^3-b^3 sin(alpha)^2+cos(alpha)^2; (15*x*y-3*x^2)/(x-5*y) 78];
simplify(m);

sin(m);
m';
diag(m);
triu(m);

%% 符号微积分

% 符号极限 limit()函数
% 用于求解函数在指定点的极限值和左右极限值
% 对于”没有定义“的极限，matlab给出的结果为NaN，对于无穷大极限给出的结果为Inf
% 求下列极限
% (1) lim_{x->a} (x^(1/m) - a^(1/m))/(x - a)
% (2) lim_{x->0} (sin(a + x) - sin(a - x))/x
syms x a m
f1 = (x^(1/m) - a^(1/m))/(x - a);
limit(f1, x, a);
limit(f1, x, a, "left");
limit(f1, x, a, 'right');

f2 = (sin(a + x) - sin(a - x))/x;
limit(f2, x, 0);

% 符号导数 diff()函数
% 求下列导数
% (1) y = sqrt(1 + exp(x));  % 求 y'
% (2) y = x*cos(x);  % 求 y'', y'''
% (3)
% x = a*cos(t);
% y = a*sin(t);
% 求 y'_x, y''_x
syms x
f1 = sqrt(1 + exp(x));
diff(f1, x, 1);

f2 = x * cos(x);
diff(f2, x, 2);
diff(f2, x, 3);

syms a t
fx = a * cos(t);
fy = a * sin(t);
yx_1 = diff(fy, t)/diff(fx, t);
yx_2 = (diff(fy, t, 2) * diff(fx, t) - diff(fy, t) * diff(fx, t, 2))/ diff(fx, t)^2;

% 符号积分 int()函数
% (1)  (3 - x^2)^3dx
% (2) (5*x*t) / (1 + x^2)dt
syms x t
f1 = (3 - x^2)^3;
int(f1, x);

f2 = (5 * x * t)/(1 + x^2);
int(f2, t);

%% 级数求和与泰勒展开的符号求解

% 无穷级数求和 symsum()函数
% (1)
% s1 = 1 + 1/4 + 1/9 + 1/16 + ... + 1/n^2 + ...
% (2)
% s2 = 1 - 1/2 + 1/3 - 1/4 + ... + (-1)^(n+1) * (1/n) + ... 
syms n
s1 = symsum(1/n^2, n, 1, inf);
s2 = symsum((-1)^(n + 1)/n, n, 1, inf);

% 函数的泰勒级数 taylor()函数
% 求 sqrt(1 - 2*x + x^3) - (1 - 3*x + x^2)^(1/3) 的 5 阶泰勒级数展开式
syms x
f = sqrt(1 - 2 * x + x^3) - (1 - 3 * x + x^2)^(1/3);
taylor(f, x, 0, 'Order', 6); % 截断阶为6， 即求前5阶泰勒展开

%% 代数方程与ODE的解析求解

% 代数方程符号求解 solve()函数
% (1) 1/(x+2) + (4*x)/(x^2 - 4) = 1 + 2/(x-2)
syms x
f = 1/(x+2) + (4*x)/(x^2 - 4) == 1 + 2/(x-2);
x = solve(f, x);

% (2)
% { 1/x^3 + 1/y^3 = 28
% { 1/x + 1/y = 4
syms x y
f1 = 1/x^3 + 1/y^3 == 28;
f2 = 1/x + 1/y == 4;
[x, y] = solve([f1, f2], [x, y]);

% 常微分方程符号求解
% 在Matlab中，用大写字母D表示导数，例如 Dy表示y'，D3y表示y'''，Dy(0)=5 表示y'(0)=5
% D3y+D2y+Dy-x+5=0 表示y''' + y" + y' - x + 5 = 0
% 符号求解函数为dsolve(e,c,v)
% e为待求解ODE，c为初值，v是个向量，用来描述方程中哪些是自变量
% 求下列微分方程的解
% (1) 求 dy/dx = (x^2 + y^2)/(2*x^2) 的通解
% (2) 求 dy/dx = x^2 / (1 + y^2) 的特解，y(2) = 1
syms x y
f1 = 'Dy = (x^2 + y^2)/(2 * x^2)';
f2 = 'Dy = x^2/(1 + y^2)';
y1 = dsolve(f1, ['x']); % 不给初值，默认求通解
y2 = dsolve(f2, 'y(2)=1', ['x']); % 给初值，求特解
