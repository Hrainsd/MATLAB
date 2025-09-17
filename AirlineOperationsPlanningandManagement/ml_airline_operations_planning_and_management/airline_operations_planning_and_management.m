%% 航空公司运行规划与管理
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% 认识intlinprog函数

% intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub)
clc;clear
f = [3, 7, -1, 1];
intcon = [1, 2, 3, 4];
A = [-2, 1, -1, 1; -1, 1, -6, -4; -5, -3, 0, -1];
B = [-1; -6; -5];
Aeq= [];
Beq = [];
lb = [0, 0, 0, 0];
ub = [1, 1, 1, 1];
x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);
disp('最优解：');
x
z = f * x;
disp('目标函数最优解：');
z

%
clc;clear
f = [-3, 2, -5];
intcon = [1, 2, 3];
A = [1, 2, -1; 1, 4, 1; 1, 2, 0; 0, 4, 1];
B = [2; 2; 2; 2];
Aeq= [];
Beq = [];
lb = [0, 0, 0];
ub = [1, 1, 1];
x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);
disp('最优解：');
x
z = f * x;
z = -z;
disp('目标函数最优解：');
z

%
clc;clear
f = zeros(1, 34);
f([1, 2, 12, 23, 24, 34]) = [8, 12, 3, 6, 10, 3];
intcon = 1:34;
A = [];
B = [];
Aeq= zeros(5, 34);
Aeq(1, [1, 2]) = [1, 1];
Aeq(2, [1, 12]) = [1, -1];
Aeq(3, [2, 12, 23, 24]) = [1, 1, -1, -1];
Aeq(4, [23, 34]) = [1, -1];
Aeq(5, [24, 34]) = [1, 1];
Beq = [1; 0; 0; 0; 1];
lb = zeros(1, 34);
ub = ones(1, 34);
x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);
disp('最优解：');
x
z = f * x;
disp('目标函数最优解：');
z

%%%%%%%%%%%%%%%%
%% 课上习题

%%
% 1.最短路径问题
% min 70x12 + 63x13 + 56x14 + 25x23 + 19x24 + 73x25 + 50x26 + 79x27 + 25x32
% + 29x34 + 69x35 + 61x36 + 19x42 + 29x43 + 67x45 + 45x46 + 85x49 + 18x56 +
% 67x57 + 69x58 + 54x59 + 87x510 + 18x65 + 72x67 + 52x68 + 51x69 + 97x610 + 17x78 +
% 31x79 + 72x710 + 17x87 + 15x89 + 31x97 + 15x98 + 69x910
clc;clear;
f = zeros(1, 910);
values = [70, 63, 56, 25, 19, 73, 50, 79, 25, 29, 69, 61, 19, ...
          29, 67, 45, 85, 18, 67, 69, 54, 87, 18, 72, 52, ...
          51, 97, 17, 31, 72, 17, 15, 31, 15, 69];
indices = [12, 13, 14, 23, 24, 25, 26, 27, 32, 34, 35, 36, ...
            42, 43, 45, 46, 49, 56, 57, 58, 59, 510, ...
            65, 67, 68, 69, 610, 78, 79, 710, 87, 89, 97, ...
            98, 910];
f(indices) = values;
intcon = 1:910;
A = [];
B = [];
Aeq = zeros(10, 910);
Aeq(1, [12, 13, 14]) = 1;
Aeq(2, [12, 32, 42, 23, 24, 25, 26, 27]) = [1, 1, 1, -1, -1, -1, -1, -1];
Aeq(3, [13, 23, 43, 32, 34, 35, 36]) = [1, 1, 1, -1, -1, -1, -1];
Aeq(4, [14, 24, 34, 42, 43, 45, 46, 49]) = [1, 1, 1, -1, -1, -1, -1, -1];
Aeq(5, [25, 35, 45, 65, 56, 57, 58, 59, 510]) = [1, 1, 1, 1, -1, -1, -1, -1, -1];
Aeq(6, [26, 36, 46, 56, 65, 67, 68, 69, 610]) = [1, 1, 1, 1, -1, -1, -1, -1, -1];
Aeq(7, [27, 57, 67, 87 ,97, 78, 79, 710]) = [1, 1, 1, 1, 1, -1, -1, -1];
Aeq(8, [58, 68, 78, 98, 87, 89]) = [1, 1, 1, 1, -1, -1];
Aeq(9, [49, 59, 69, 79, 89, 97, 98, 910]) = [1, 1, 1, 1, 1, -1, -1, -1];
Aeq(10, [510, 610, 710, 910]) = [1, 1, 1, 1];
Beq = [1; 0; 0; 0; 0; 0; 0; 0; 0; 1];
lb = zeros(1, 910);
ub = ones(1, 910);

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 2.最小成本流问题
% min 5x13 + 8x14 + 7x23 + 4x24 +x35 + 5x36 + 8x37 + 3x45 + 4x46 + 4x47
clc;clear;
f = zeros(1, 47);
f([13, 14, 23, 24, 35, 36, 37, 45, 46, 47]) = [5, 8, 7, 4, 1, 5, 8, 3, 4, 4];
intcon = 1:47;
A = [];
B = [];
Aeq = zeros(7, 47);
Aeq(1, [13, 14]) = 1;
Aeq(2, [23, 24]) = 1;
Aeq(3, [13, 23, 35, 36, 37]) = [1, 1, -1, -1, -1];
Aeq(4, [14, 24, 45, 46, 47]) = [1, 1, -1, -1, -1];
Aeq(5, [35, 45]) = 1;
Aeq(6, [36, 46]) = 1;
Aeq(7, [37, 47]) = 1;
Beq = [75; 75; 0; 0; 50; 60; 40];
lb = zeros(1, 47);
ub = zeros(1, 47);
ub([13, 14, 23, 24, 35, 36, 37, 45, 46, 47]) = [75, 50, 75, 50, 150, 150, 150, 50, 50, 50];

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 3.最大流问题
% max x12(或x35 + x45)
% min -x12
clc;clear;
f = zeros(1, 45);
f(12) = -1;
intcon = 1:45;
A = [];
B = [];
Aeq = zeros(3, 45);
Aeq(1, [12, 23, 24]) = [1, -1, -1];
Aeq(2, [23, 35]) = [1, -1];
Aeq(3, [24, 45]) = [1, -1];
Beq = zeros(3, 1);
lb = zeros(1, 45);
ub = zeros(1, 45);
ub([12, 23, 24, 35, 45]) = [3, 2, 3, 1, 2];

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x;
z = -z

%%
% 4.多商品网络流问题
% min 5x131 + 5x132 + 8x141 + 8x142 + 7x231 + 7x232 + 4x241 + 4x242 +x351 +
% x352 + 5x361 + 5x362 + 8x371 + 8x372 + 3x451 + 3x452 + 4x461 + 4x462 +
% 4x471 + 4x472
clc;clear
f = zeros(1, 472);
f([131, 132, 141, 142, 231, 232, 241, 242, 351, 352, 361, 362, 371, 372, 451, 452, 461, 462, 471, 472]) = [5, 5, 8, 8, 7, 7, 4, 4, 1, 1, 5, 5, 8, 8, 3, 3, 4, 4, 4, 4];
intcon = 1:472;
A = zeros(5, 472);
A(1, [141, 142]) = 1;
A(2, [241, 242]) = 1;
A(3, [451, 452]) = 1;
A(4, [461, 462]) = 1;
A(5, [471, 472]) = 1;
B = 50 * ones(5, 1);
Aeq = zeros(14, 472);
Aeq(1, [131, 141]) = 1;
Aeq(2, [231, 241]) = 1;
Aeq(3, [131, 231, 351, 361, 371]) = [1, 1, -1, -1, -1];
Aeq(4, [141, 241, 451, 461, 471]) = [1, 1, -1, -1, -1];
Aeq(5, [351, 451]) = 1;
Aeq(6, [361, 461]) = 1;
Aeq(7, [371, 471]) = 1;
Aeq(8, [132, 142]) = 1;
Aeq(9, [232, 242]) = 1;
Aeq(10, [132, 232, 352, 362, 372]) = [1, 1, -1, -1, -1];
Aeq(11, [142, 242, 452, 462, 472]) = [1, 1, -1, -1, -1];
Aeq(12, [352, 452]) = 1;
Aeq(13, [362, 462]) = 1;
Aeq(14, [372, 472]) = 1;
Beq = [40; 50; 0; 0; 30; 30; 30; 35; 25; 0; 0; 20; 30; 10];
lb = zeros(1, 472);
ub = zeros(1, 472);
ub([131, 141, 231, 241, 351, 361, 371, 451, 461, 471, 132, 142, 232, 242, 352, 362, 372, 452, 462, 472]) = [40, 40, 50, 50, 90, 90, 90, 90, 90, 90, 35, 35, 25, 25, 60, 60, 60, 60, 60, 60];

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 5.集覆盖和集划分问题
% min x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12
clc;clear
x = [0    1037   674   1398   789   2182   479   841   687   1878   2496   2618;
     1037  0      1005  1949   1804  2979   1507  222   574   2343   3095   2976;
     674   1005   0     1008   1067  2054   912   802   452   1390   2142   2013;
     1398  1949   1008  0      1019  1059   1273  1771  1411  504    1235   1307;
     789   1804   1067  1019   0     1538   356   1608  1313  1438   1912   2274;
     2182  2979   2054  1059   1538  0      1883  2786  2426  715    379    1131;
     479   1507   912   1273   356   1883   0     1311  1070  1738   2249   2574;
     841   222    802   1771   1608  2786   1311  0     368   2182   2934   2815;
     687   574    452   1411   1313  2426   1070  368   0     1826   2578   2465;
     1878  2343   1390  504    1438  715    1738  2182  1826  0      752    836;
     2496  3095   2142  1235   1912  379    2249  2934  2578  752    0      808;
     2618  2976   2013  1307   2274  1131   2574  2815  2465  836    808    0];
x(find(x <= 1000)) = 1;
x(find(x >= 1000)) = 0;
% x = [1     0     1     0     1     0     1     1     1     0     0     0;
%      0     1     0     0     0     0     0     1     1     0     0     0;
%      1     0     1     0     0     0     1     1     1     0     0     0;
%      0     0     0     1     0     0     0     0     0     1     0     0;
%      1     0     0     0     1     0     1     0     0     0     0     0;
%      0     0     0     0     0     1     0     0     0     1     1     0;
%      1     0     1     0     1     0     1     0     0     0     0     0;
%      1     1     1     0     0     0     0     1     1     0     0     0;
%      1     1     1     0     0     0     0     1     1     0     0     0;
%      0     0     0     1     0     1     0     0     0     1     1     1;
%      0     0     0     0     0     1     0     0     0     1     1     1;
%      0     0     0     0     0     0     0     0     0     1     1     1];
f = ones(1, 12);
intcon = 1:12;
A = [];
B = [];
Aeq = zeros(12, 12);
Aeq(1, [1, 3, 5, 7, 8, 9]) = 1;
Aeq(2, [2, 8 , 9]) = 1;
Aeq(3, [1, 3, 7 ,8, 9]) = 1;
Aeq(4, [4, 10]) = 1;
Aeq(5, [1, 5 ,7]) = 1;
Aeq(6, [6, 10, 11]) = 1;
Aeq(7, [1, 3, 5, 7]) = 1;
Aeq(8, [1, 2, 3, 8, 9]) = 1;
Aeq(9, [1, 2, 3, 8, 9]) = 1;
Aeq(10, [4, 6, 10, 11, 12]) = 1;
Aeq(11, [6, 10, 11, 12]) = 1;
Aeq(12, [10, 11, 12]) = 1;
Beq = ones(12, 1);
lb = zeros(1, 12);
ub = ones(1, 12);
x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 6.航班串遍历问题
% min 702x12 + 454x13 + 842x14 + 2396x15 + 1196x16 + 864x17 +
%     702x21 + 324x23 + 1093x24 + 2136x25 + 764x26 + 845x27 +
%     454x31 + 324x32 + 1137x34 + 2180x35 + 798x36 + 664x37 +
%     842x41 + 1093x42 + 1137x43 + 1616x45 + 1857x46 + 1706x47 +
%     2396x51 + 2136x52 + 2180x53 + 1616x54 + 2900x56 + 2844x57 +
%     1196x61 + 764x62 + 798x63 + 1857x64 + 2900x65 + 396x67 +
%     864x71 + 845x72 + 664x73 + 1706x74 + 2844x75 + 396x76
clc;clear
f = zeros(1, 77);
f([11, 22, 33, 44, 55, 66, 77]) = 1000000;
f([12, 13, 14, 15, 16, 17]) = [702, 454, 842, 2396, 1196, 864];
f([21, 23, 24, 25, 26, 27]) = [702, 324, 1093, 2136, 764, 845];
f([31, 32, 34, 35, 36, 37]) = [454, 324, 1137, 2180, 798, 664];
f([41, 42, 43, 45, 46, 47]) = [842, 1093, 1137, 1616, 1857, 1706];
f([51, 52, 53, 54, 56, 57]) = [2396, 2136, 2180, 1616, 2900, 2844];
f([61, 62, 63, 64, 65, 67]) = [1196, 764, 798, 1857, 2900, 396];
f([71, 72, 73, 74, 75, 76]) = [864, 845, 664, 1706, 2844, 396];
intcon = 1:77;
A = zeros(2, 77);
A(1, [45 ,54]) = 1;
A(2, [67, 76]) = 1;
B = ones(2, 1);
Aeq = zeros(14, 77);
Aeq(1, [11, 12, 13, 14, 15, 16, 17]) = 1;
Aeq(2, [21, 22, 23, 24, 25, 26, 27]) = 1;
Aeq(3, [31, 32, 33, 34, 35, 36, 37]) = 1;
Aeq(4, [41, 42, 43, 44, 45, 46, 47]) = 1;
Aeq(5, [51, 52, 53, 54, 55, 56, 57]) = 1;
Aeq(6, [61, 62, 63, 64, 65, 66, 67]) = 1;
Aeq(7, [71, 72, 73, 74, 75, 76, 77]) = 1;
Aeq(8, [11, 21, 31, 41, 51, 61, 71]) = 1;
Aeq(9, [12, 22, 32, 42, 52, 62, 72]) = 1;
Aeq(10, [13, 23, 33, 43, 53, 63, 73]) = 1;
Aeq(11, [14, 24, 34, 44, 54, 64, 74]) = 1;
Aeq(12, [15, 25, 35, 45, 55, 65, 75]) = 1;
Aeq(13, [16, 26, 36, 46, 56, 66, 76]) = 1;
Aeq(14, [17, 27, 37, 47, 57, 67, 77]) = 1;
Beq = ones(14, 1);
lb = zeros(1, 77);
ub = ones(1, 77);

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 7.机型分配
clc;clear
% 航线距离 平均值 标准差
infm = [
 2475  175   35
 2475  182   36
 2475  145   29
 2586  178   35
 2586  195   39
 2586  162   32
  740  165   33
  740  182   36
  740  170   34
  760  191   38
  760  171   34
  760  165   33
 1090  198   39
 1090  182   36
 1090  168   33
  187  115   23
  187  146   29
  187  120   24
  228  135   27
  228  109   21
  228   98   19
 2475  150   30
 2475  145   29
 2475  125   25
 2586  148   29
 2586  138   27
 2586  121   24
  740  132   26
  740  129   25
  740  117   23
  760  168   33
  760  160   32
  760  191   38
 1090  165   33
 1090  184   36
 1090  192   38
  187  147   29
  187  135   27
  187  146   29
  228  105   21
  228  115   23
  228  118   23];

% 参数
aircraft_seats = [162, 200];
casms = [0.042, 0.044];
rasms = [0.15, 0.15];
recaps = [0.15, 0.15];
cost1 = zeros(42, 2);
cost2 = zeros(42, 2);
cost = zeros(42, 2);
% fun = @(x) (x - seat) .* 1/(sqrt(2 * pi) * infm(j, 3)) .* exp(-(x - infm(j, 2)).^2/(2 * infm(j, 3).^2));
% exp: integral(@(x) (x-162) .* 1/(sqrt(2 * pi) * 30) .* exp(-(x - 150).^2/(2 * 30.^2)), 162, + inf)
% exp: integral(@(x) (x-200) .* 1/(sqrt(2 * pi) * 30) .* exp(-(x - 150).^2/(2 * 30.^2)), 200, + inf)
% round(round(integral(@(x) (x-200) .* 1/(sqrt(2 * pi) * 30) .* exp(-(x - 150).^2/(2 * 30.^2)), 200, + inf), 2) * 0.15 * 2475, 2)

% 制作f
for i = 1:2
    seat = aircraft_seats(i);
    casm = casms(i);
    rasm = rasms(i);
    recap = recaps(i);
    for j = 1:42
        cost1(j, i) = round(casm * infm(j, 1) * seat, 2);
        cost2(j, i) = round(rasm * infm(j, 1) * round(integral(@(x) (x - seat) .* 1/(sqrt(2 * pi) * infm(j, 3)) .* exp(-(x - infm(j, 2)).^2/(2 * infm(j, 3).^2)), seat, inf), 2), 2);
        cost(j, i) = round(cost1(j, i) + round((1 - recap) * cost2(j, i), 2), 2);
    end
end

cost = cost';
f_values = reshape(cost, 1, []);
f_ind = [];
for i = 1:42  % 控制第一位数字
    for j = 1:2  % 控制第二位数字
        f_ind = [f_ind, i*10 + j];
    end
end
f = zeros(1, 1262);
f(f_ind) = f_values;

intcon = 1:1:1262;
A = zeros(2, 1262);

% 机队规模约束
A(1, [481, 541, 601, 661, 721, 781, 841, 1261]) = 1;
A(2, [482, 542, 602, 662, 721, 781, 841, 1262]) = 1;
B = [9; 6];

% 航班覆盖约束
Aeq = zeros(210, 1262);
f_ind1 = reshape(f_ind, 2, []);
for i = 1:42
    Aeq(i, [f_ind1(:, i)]) = 1;
end

% 飞机平衡约束
% 制作第一列变量
f_ind2 = [];
for i = 43:126  % 控制第一位数字
    for j = 1:2  % 控制第二位数字
        f_ind2 = [f_ind2, i*10 + j];
    end
end
ind_1 = reshape(f_ind2, 2, [])';

% 制作第二列变量
ind_2_1 = ind_1(1:42, :);
% 每六个数一组
numGroups = size(ind_2_1, 1) / 6;
% 处理每组
for i = 1:numGroups
    groupIdx = (i-1)*6 + 1:i*6;
    ind_2_1(groupIdx, :) = [ind_2_1(groupIdx(end), :); ind_2_1(groupIdx(1:end-1), :)];
end
ind_2_2 = ind_1(43:end, :);
ind_2_2 = [ind_2_2(end, :); ind_2_2(1:end-1, :)];
ind_2 = [ind_2_1; ind_2_2];

% 制作第三列变量
ind_3 = [
    11; 21; 221; 31; 231; 241;
    41; 251; 51; 61; 261; 271;
    161; 371; 171; 381; 181; 391;
    71; 281; 81; 291; 91; 301;
    101; 311; 111; 321; 121; 331;
    401; 191; 411; 201; 421; 211;
    131; 341; 141; 351; 361; 151;
    11; 41; 161; 401; 251; 71;
    221; 371; 101; 191; 131; 311;
    21; 51; 171; 281; 341; 411; 
    81; 381; 111; 201; 141; 321;
    181; 291; 351; 421; 31; 61; 
    261; 231; 91; 121; 331; 361;
    151; 211; 241; 271; 301; 391];

% 合并三列变量
ind = [ind_1, ind_2, ind_3];

% 制作变量系数
coef = ones(84, 3);
coef(:, 2) = -1;
coef(:, 3) = [1; 1; -1; 1; -1; -1;
              1; -1; 1; 1; -1; -1;
              1; -1; 1; -1; 1; -1;
              1; -1; 1; -1; 1; -1;
              1; -1; 1; -1; 1; -1;
              -1; 1; -1; 1; -1; 1;
              1; -1; 1; -1; -1; 1;
              -1; -1; -1; 1; 1; -1; 1; 1; -1; -1; -1; 1; -1; -1; -1; 1; 1; 1; -1; 1; -1; -1; -1; 1; -1; 1; 1; 1; -1; -1; 1; 1; -1; -1; 1; 1; -1; -1; 1; 1; 1; 1];
for i = 1:2
    for j = 1:84
        Aeq(j+ (i-1) * 84 + 42, [ind(j, i), ind(j, i + 2), ind(j, 5)]) = coef(j, :);
    end
end

% 制作飞机平衡约束矩阵
BM = [];

% 从第43行开始，逐行找到不等于0的列索引
for i = 43:size(Aeq, 1)
    % 找到当前行中所有不为0的列索引
    nonZeroCols = find(Aeq(i, :) ~= 0);
    
    % 将不为0的列索引添加到BM中
    BM = [BM; nonZeroCols];
end

BMCoef = [coef(:,3);coef(:,3)];
BM = [BM(:,2),-BM(:,3),BM(:,1).*BMCoef];

Beq = zeros(210, 1);
Beq([1:1:42]) = 1;

lb = zeros(1, 1262);
ub = ones(1, 1262);
ub(ind_1(:, 1)) = 9;
ub(ind_1(:, 2)) = 6;

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 8.飞机航线调配

% B757-200
clc; clear;
FlightMatrix = {
    125, 'JFK', '07:25', 'SFO', '09:55', 5.5;
    110, 'ATL', '08:10', 'JFK', '10:40', 2.5;
    113, 'MIA', '09:10', 'JFK', '12:10', 3;
    131, 'JFK', '09:30', 'ATL', '12:00', 2.5;
    105, 'SFO', '09:50', 'JFK', '18:20', 5.5;
    138, 'JFK', '12:30', 'BOS', '14:00', 1.5;
    111, 'ATL', '13:10', 'JFK', '15:40', 2.5;
    114, 'MIA', '14:30', 'JFK', '17:30', 3;
    118, 'BOS', '15:00', 'JFK', '16:30', 1.5;
    135, 'JFK', '15:10', 'MIA', '18:10', 3;
    133, 'JFK', '18:05', 'ATL', '20:35', 2.5;
    136, 'JFK', '18:10', 'MIA', '21:10', 3
};

Flight_len = size(FlightMatrix, 1);

% 获取起飞时间和到达时间列
DepartureTimes = FlightMatrix(:, 3);
ArrivalTimes = FlightMatrix(:, 5);

% 初始化分钟数组
DepartureMinutes = zeros(size(DepartureTimes));
ArrivalMinutes = zeros(size(ArrivalTimes));

% 转换时间为分钟数
for i = 1:length(DepartureTimes)
    % 转换起飞时间
    depTime = datetime(DepartureTimes{i}, 'InputFormat', 'HH:mm');
    DepartureMinutes(i) = hour(depTime) * 60 + minute(depTime);
    
    % 转换到达时间
    arrTime = datetime(ArrivalTimes{i}, 'InputFormat', 'HH:mm');
    ArrivalMinutes(i) = hour(arrTime) * 60 + minute(arrTime);
end

FlightChains = {};  % 用于存储航线

% 查找航线
for i = 1:Flight_len
    
    for j = 1:Flight_len
        current_chain1 = FlightMatrix(i, :);
        current_chain_length1 = 1;
        current_arrival_airport = FlightMatrix{i, 4};
        current_arrival_minute = ArrivalMinutes(i);
        next_departure_airport = FlightMatrix{j, 2};
        next_departure_minute = DepartureMinutes(j);
        
        % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
        if strcmp(current_arrival_airport, next_departure_airport) && ...
                (next_departure_minute - current_arrival_minute) >= 45
            % 将该航班加入航线
            current_chain1_2 = [current_chain1; FlightMatrix(j, :)];
            current_chain_length1_2 = current_chain_length1 + 1;
        else
            FlightChains{end+1} = current_chain1;
            continue
        end

        for k = 1:Flight_len
            current_chain2 = current_chain1_2;
            current_chain_length2 = current_chain_length1_2;
            current_arrival_airport = FlightMatrix{j, 4};
            current_arrival_minute = ArrivalMinutes(j);
            next_departure_airport = FlightMatrix{k, 2};
            next_departure_minute = DepartureMinutes(k);
                
            % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
            if strcmp(current_arrival_airport, next_departure_airport) && ...
                    (next_departure_minute - current_arrival_minute) >= 45
                % 将该航班加入航线
                current_chain1_2_3 = [current_chain2; FlightMatrix(k, :)];
                current_chain_length1_2_3 = current_chain_length2 + 1;
            else
                % 保存到 FlightChains
                FlightChains{end+1} = current_chain2;
                continue
            end

            for m = 1:Flight_len
                current_chain3 = current_chain1_2_3;
                current_chain_length3 = current_chain_length1_2_3;
                current_arrival_airport = FlightMatrix{k, 4};
                current_arrival_minute = ArrivalMinutes(k);
                next_departure_airport = FlightMatrix{m, 2};
                next_departure_minute = DepartureMinutes(m);
                    
                % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
                if strcmp(current_arrival_airport, next_departure_airport) && ...
                        (next_departure_minute - current_arrival_minute) >= 45
                    % 将该航班加入航线
                    current_chain1_2_3_4 = [current_chain3; FlightMatrix(m, :)];
                    current_chain_length1_2_3_4 = current_chain_length3 + 1;
                else
                        FlightChains{end+1} = current_chain3;
                    continue
                end
                for p = 1:Flight_len
                    current_chain4 = current_chain1_2_3_4;
                    current_chain_length4 = current_chain_length1_2_3_4;
                    current_arrival_airport = FlightMatrix{m, 4};
                    current_arrival_minute = ArrivalMinutes(m);
                    next_departure_airport = FlightMatrix{p, 2};
                    next_departure_minute = DepartureMinutes(p);
                        
                    % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
                    if strcmp(current_arrival_airport, next_departure_airport) && ...
                            (next_departure_minute - current_arrival_minute) >= 45
                        % 将该航班加入航线
                        current_chain1_2_3_4_5 = [current_chain4; FlightMatrix(p, :)];
                        current_chain_length1_2_3_4_5 = current_chain_length4 + 1;
                    else
                            FlightChains{end+1} = current_chain4;
                        continue
                    end

                    for n = 1:Flight_len
                        current_chain5 = current_chain1_2_3_4_5;
                        current_chain_length5 = current_chain_length1_2_3_4_5;
                        current_arrival_airport = FlightMatrix{p, 4};
                        current_arrival_minute = ArrivalMinutes(p);
                        next_departure_airport = FlightMatrix{n, 2};
                        next_departure_minute = DepartureMinutes(n);
                            
                        % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
                        if strcmp(current_arrival_airport, next_departure_airport) && ...
                                (next_departure_minute - current_arrival_minute) >= 45
                            % 将该航班加入航线
                            current_chain6 = [current_chain1_2_3_4_5; FlightMatrix(n, :)];
                            current_chain_length6 = current_chain_length1_2_3_4_5 + 1;
                            FlightChains{end+1} = current_chain6;
                        else
                            FlightChains{end+1} = current_chain5;
                            continue
                        end
                    end
                end
            end
        end
    end
end

% 显示结果
disp(FlightChains);

% 去除重复航班链
FlightChainsStr = cellfun(@(x) strjoin(cellfun(@num2str, x, 'UniformOutput', false), ','), FlightChains, 'UniformOutput', false); % 'UniformOutput', false允许返回不同大小的输出，这样可以生成一个元胞数组
[~, uniqueIdx] = unique(FlightChainsStr);
UniqueFlightChains = FlightChains(uniqueIdx);
disp(UniqueFlightChains);

% 一日航班结果
% 初始化结果矩阵，预分配最大可能的大小以提高效率
MaxChainLength = max(cellfun('size', UniqueFlightChains, 1));
ResultMatrix_flight_1d = zeros(length(UniqueFlightChains), MaxChainLength);

% 将 UniqueFlightChains 的内容转换为矩阵
for i = 1:length(UniqueFlightChains)
    currentChain = UniqueFlightChains{i}(:, 1);
    ResultMatrix_flight_1d(i, 1:length(currentChain)) = vertcat(currentChain{:});
end

% 显示结果矩阵
disp(ResultMatrix_flight_1d);

% 第一日过夜机场结果
ResultMatrix_airport_1d = {};
for i = 1:length(UniqueFlightChains)
    currentairport = UniqueFlightChains{i}(end, 4);
    ResultMatrix_airport_1d{end+1} = currentairport;
end

% 显示矩阵
disp(ResultMatrix_airport_1d);

% 连接第二日航班
ResultMatrix_filght_1d_connections = [];
for i = 1:length(UniqueFlightChains)
    airport = ResultMatrix_airport_1d{i};
    currentRow = [];
    for j = 1:Flight_len  % 遍历所有航班
        if strcmp(FlightMatrix{j, 2}, airport)
            currentRow(end+1) = FlightMatrix{j, 1};
        end
    end
    ResultMatrix_filght_1d_connections(i, 1:length(currentRow)) = currentRow;
end

% 显示结果矩阵
disp(ResultMatrix_filght_1d_connections);

ResultMatrix_flight_2d = [];
for i = 1:length(UniqueFlightChains)
    for j = 1:size(ResultMatrix_filght_1d_connections, 2)
        if ResultMatrix_filght_1d_connections(i, j) ~= 0
            % 查找匹配的航班号的行
            matchingRows = find(ResultMatrix_flight_1d(:, 1) == ResultMatrix_filght_1d_connections(i, j));
            for k = 1:length(matchingRows)
                % 获取当前匹配行
                currentRow = ResultMatrix_flight_1d(matchingRows(k), :);
                % 将当前航班链与匹配行合并
                newRow = [ResultMatrix_flight_1d(i, :), currentRow];
                ResultMatrix_flight_2d(end + 1, 1:length(newRow)) = newRow;  % 确保行数一致
            end
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_2d);

% 第二日航班结果
ResultMatrix_filght_2d_last = zeros(size(ResultMatrix_flight_2d, 1), 1);

% 遍历 ResultMatrix2 的每一行
for i = 1:size(ResultMatrix_flight_2d, 1)
    % 获取当前行
    currentRow = ResultMatrix_flight_2d(i, :);
    % 找到最后一个不为0的元素
    lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
    % 存储最后一个不为0的元素
    ResultMatrix_filght_2d_last(i) = currentRow(lastNonZeroIdx);
end

% 显示结果矩阵
disp(ResultMatrix_filght_2d_last);

% 第二日最后机场结果
ResultMatrix_airport_2d = {};
for i = 1:length(ResultMatrix_filght_2d_last)
    for j = 1:size(FlightMatrix, 1)
        if ResultMatrix_filght_2d_last(i) == FlightMatrix{j, 1}
            ResultMatrix_airport_2d{end+1} = FlightMatrix{j, 4};
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_2d);

% 连接第三日航班
ResultMatrix_flight_2d_connections = [];
for i = 1:length(ResultMatrix_flight_2d)
    airport2 = ResultMatrix_airport_2d{i};
    currentRow = [];
    for j = 1:Flight_len  % 遍历所有航班
        if strcmp(FlightMatrix{j, 2}, airport2)
            currentRow(end+1) = FlightMatrix{j, 1};
        end
    end
    ResultMatrix_flight_2d_connections(i, 1:length(currentRow)) = currentRow;
end

% 显示结果矩阵
disp(ResultMatrix_flight_2d_connections);

ResultMatrix_flight_3d = [];
for i = 1:length(ResultMatrix_flight_2d)
    for j = 1:size(ResultMatrix_flight_2d_connections, 2)
        if ResultMatrix_flight_2d_connections(i, j) ~= 0
            % 查找匹配的航班号的行
            matchingRows = find(ResultMatrix_flight_1d(:, 1) == ResultMatrix_flight_2d_connections(i, j));
            for k = 1:length(matchingRows)
                % 获取当前匹配行
                currentRow = ResultMatrix_flight_1d(matchingRows(k), :);
                % 将当前航班链与匹配行合并
                newRow = [ResultMatrix_flight_2d(i, :), currentRow];
                ResultMatrix_flight_3d(end + 1, 1:length(newRow)) = newRow;  % 确保行数一致
            end
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_3d);

maxnum_1d = size(ResultMatrix_flight_3d, 2)/3;
ResultMatrix_flight_3d_last3 = [];
for i = 1:3
    for j = 1:size(ResultMatrix_flight_3d, 1)
        % 获取当前行
        currentRow = ResultMatrix_flight_3d(j, maxnum_1d * (i - 1) + 1:maxnum_1d * i);
        % 找到最后一个不为0的元素
        lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
        % 存储最后一个不为0的元素
        ResultMatrix_flight_3d_last3(j, i) = currentRow(lastNonZeroIdx);
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_3d_last3);

% JFK机场停场过夜条件
% 不同的FlightMatrix，需要改为不同的所有在JFK机场过夜的航班集合
ResultMatrix_flight_3d_final1 = ResultMatrix_flight_3d;

% 找到第四列为'JFK'的行
rows = strcmp(FlightMatrix(:, 4), 'JFK');

% 提取航班号
excludedFlights = cell2mat(FlightMatrix(rows, 1));

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = true(size(ResultMatrix_flight_3d_final1, 1), 1);

for i = 1:size(ResultMatrix_flight_3d_final1, 1)
    n = 0;  % 计数器
    for j = 1:3
        if ~ismember(ResultMatrix_flight_3d_last3(i, j), excludedFlights)  % 检查是否在排除列表中
            n = n + 1;
        end
    end
    if n == 3
        rowsToKeep(i) = false;  % 标记该行需要被删除
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_3d_final1 = ResultMatrix_flight_3d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_3d_final1);

% 起始终止机场条件
ResultMatrix_flight_3d_final1_last3 = [];

for i = 1:3
    for j = 1:size(ResultMatrix_flight_3d_final1, 1)
        % 获取当前行
        currentRow = ResultMatrix_flight_3d_final1(j, maxnum_1d * (i - 1) + 1:maxnum_1d * i);
        % 找到最后一个不为0的元素
        lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
        % 存储最后一个不为0的元素
        ResultMatrix_flight_3d_final1_last3(j, i) = currentRow(lastNonZeroIdx);
    end
end

ResultMatrix_flight_1d_first_last = [ResultMatrix_flight_3d_final1(:, 1), ResultMatrix_flight_3d_final1_last3(:, 3)];
ResultMatrix_airport_3d_first_last = cell(size(ResultMatrix_flight_1d_first_last, 1), 2);

for i = 1:size(ResultMatrix_flight_1d_first_last, 1)
    % 查找起始机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 1) == FlightMatrix{j, 1}
            ResultMatrix_airport_3d_first_last{i, 1} = FlightMatrix{j, 2};  % 起飞机场
            break;  % 找到后跳出循环
        end
    end
    
    % 查找终止机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 2) == FlightMatrix{j, 1}
            ResultMatrix_airport_3d_first_last{i, 2} = FlightMatrix{j, 4};  % 到达机场
            break;  % 找到后跳出循环
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_3d_first_last);

ResultMatrix_flight_1d_final1 = ResultMatrix_flight_3d_final1;

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = true(size(ResultMatrix_flight_1d_final1, 1), 1);

for i = 1:size(ResultMatrix_flight_1d_final1, 1)
    if ~strcmp(ResultMatrix_airport_3d_first_last{i, 1}, ResultMatrix_airport_3d_first_last{i, 2})
        rowsToKeep(i) = false;  % 标记该行需要被删除
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_1d_final1 = ResultMatrix_flight_1d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_1d_final1);

% 添加维护机会列
ResultMatrix_flight_3d_final_last3 = [];
for i = 1:3
    for j = 1:size(ResultMatrix_flight_1d_final1, 1)
        % 获取当前行
        currentRow = ResultMatrix_flight_1d_final1(j, maxnum_1d * (i - 1) + 1:maxnum_1d * i);
        % 找到最后一个不为0的元素
        lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
        % 存储最后一个不为0的元素
        ResultMatrix_flight_3d_final_last3(j, i) = currentRow(lastNonZeroIdx);
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_3d_final_last3);

MaintenanceOpp = [];
for i = 1:size(ResultMatrix_flight_3d_final_last3, 1)
    n = 0;
    for j = 1:3
        if ismember(ResultMatrix_flight_3d_final_last3(i, j), excludedFlights)
            n = n + 1;
        end
    end
    MaintenanceOpp(i, 1) = n;
end

% 显示结果矩阵
disp(MaintenanceOpp);

ResultMatrix = [ResultMatrix_flight_1d_final1, MaintenanceOpp];

% 显示结果矩阵
disp(ResultMatrix);

% 优化
VarNum = size(ResultMatrix, 1);
Aeq_size = 3 * Flight_len;

% 航班覆盖
FlightCover = {};
for i = 1:Flight_len
    CurrentRow = []; % 每次循环重置
    for j = 1:3
        n = 0;
        for k = 1:VarNum
            if ismember(FlightMatrix{i, 1}, ResultMatrix(k, maxnum_1d * (j - 1) + 1:maxnum_1d * j))
                n = n + 1;
                CurrentRow(j, n) = k;
            end
        end
    end
    FlightCover = [FlightCover; {CurrentRow}];
end

% 显示结果矩阵
disp(FlightCover);

f = -ResultMatrix(:, end)';
intcon = 1:1:VarNum;
A = ones(1, VarNum);
B = 8; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%根据需要修改
Aeq = zeros(Aeq_size, VarNum);
Beq = ones(Aeq_size, 1);
lb = zeros(1, VarNum);
ub = ones(1, VarNum);

n = 0;
for i = 1:Flight_len
    for j = 1:3
        n = n + 1;
        
        % 确保 FlightCover{i}(j, :) 是一个有效的索引数组
        indices = FlightCover{i}(j, :);
        indices = indices(indices > 0); % 排除非正整数的情况
        
        % 如果 indices 不为空，则将 Aeq 中这些位置设为 1
        if ~isempty(indices)
            Aeq(n, indices) = 1;
        end
    end
end

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x;
z = -z



% B737-800
clc; clear;
FlightMatrix = {
    101, 'LAX', '05:00', 'JFK', '13:30', 5.5;
    104, 'SFO', '05:05', 'JFK', '13:35', 5.5;
    116, 'BOS', '06:15', 'JFK', '07:45', 1.5;
    140, 'JFK', '06:20', 'IAD', '07:20', 1;
    107, 'ORD', '07:30', 'JFK', '10:30', 2;
    122, 'JFK', '07:35', 'LAX', '10:05', 5.5;
    137, 'JFK', '07:40', 'BOS', '09:10', 1.5;
    119, 'IAD', '08:15', 'JFK', '09:15', 1;
    102, 'LAX', '09:45', 'JFK', '18:15', 5.5;
    117, 'BOS', '10:00', 'JFK', '11:30', 1.5;
    128, 'JFK', '10:05', 'ORD', '11:05', 2;
    134, 'JFK', '10:35', 'MIA', '13:35', 3;
    141, 'JFK', '12:00', 'IAD', '13:00', 1;
    108, 'ORD', '12:20', 'JFK', '15:20', 2;
    120, 'IAD', '14:25', 'JFK', '15:25', 1;
    132, 'JFK', '14:35', 'ATL', '17:35', 2.5;
    129, 'JFK', '15:05', 'ORD', '16:05', 2;
    142, 'JFK', '15:15', 'IAD', '16:15', 1;
    103, 'LAX', '15:20', 'JFK', '23:50', 5.5;
    106, 'SFO', '15:25', 'JFK', '23:55', 5.5;
    126, 'JFK', '15:30', 'SFO', '18:00', 5.5;
    123, 'JFK', '16:00', 'LAX', '18:30', 5.5;
    109, 'ORD', '17:10', 'JFK', '20:10', 2;
    112, 'ATL', '18:00', 'JFK', '20:30', 2.5;
    115, 'MIA', '18:15', 'JFK', '21:15', 3;
    121, 'IAD', '18:30', 'JFK', '19:30', 1;
    124, 'JFK', '19:00', 'LAX', '21:30', 5.5;
    127, 'JFK', '20:00', 'SFO', '22:30', 5.5;
    130, 'JFK', '21:00', 'ORD', '22:00', 2;
    139, 'JFK', '21:30', 'BOS', '23:00', 1.5
};

Flight_len = size(FlightMatrix, 1);

% 获取起飞时间和到达时间列
DepartureTimes = FlightMatrix(:, 3);
ArrivalTimes = FlightMatrix(:, 5);

% 初始化分钟数组
DepartureMinutes = zeros(size(DepartureTimes));
ArrivalMinutes = zeros(size(ArrivalTimes));

% 转换时间为分钟数
for i = 1:length(DepartureTimes)
    % 转换起飞时间
    depTime = datetime(DepartureTimes{i}, 'InputFormat', 'HH:mm');
    DepartureMinutes(i) = hour(depTime) * 60 + minute(depTime);
    
    % 转换到达时间
    arrTime = datetime(ArrivalTimes{i}, 'InputFormat', 'HH:mm');
    ArrivalMinutes(i) = hour(arrTime) * 60 + minute(arrTime);
end

FlightChains = {};  % 用于存储航线

% 查找航线
for i = 1:Flight_len
    
    for j = 1:Flight_len
        current_chain1 = FlightMatrix(i, :);
        current_chain_length1 = 1;
        current_arrival_airport = FlightMatrix{i, 4};
        current_arrival_minute = ArrivalMinutes(i);
        next_departure_airport = FlightMatrix{j, 2};
        next_departure_minute = DepartureMinutes(j);
        
        % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
        if strcmp(current_arrival_airport, next_departure_airport) && ...
                (next_departure_minute - current_arrival_minute) >= 45
            % 将该航班加入航线
            current_chain1_2 = [current_chain1; FlightMatrix(j, :)];
            current_chain_length1_2 = current_chain_length1 + 1;
        else
            FlightChains{end+1} = current_chain1;
            continue
        end

        for k = 1:Flight_len
            current_chain2 = current_chain1_2;
            current_chain_length2 = current_chain_length1_2;
            current_arrival_airport = FlightMatrix{j, 4};
            current_arrival_minute = ArrivalMinutes(j);
            next_departure_airport = FlightMatrix{k, 2};
            next_departure_minute = DepartureMinutes(k);
                
            % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
            if strcmp(current_arrival_airport, next_departure_airport) && ...
                    (next_departure_minute - current_arrival_minute) >= 45
                % 将该航班加入航线
                current_chain1_2_3 = [current_chain2; FlightMatrix(k, :)];
                current_chain_length1_2_3 = current_chain_length2 + 1;
            else
                % 保存到 FlightChains
                FlightChains{end+1} = current_chain2;
                continue
            end

            for m = 1:Flight_len
                current_chain3 = current_chain1_2_3;
                current_chain_length3 = current_chain_length1_2_3;
                current_arrival_airport = FlightMatrix{k, 4};
                current_arrival_minute = ArrivalMinutes(k);
                next_departure_airport = FlightMatrix{m, 2};
                next_departure_minute = DepartureMinutes(m);
                    
                % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
                if strcmp(current_arrival_airport, next_departure_airport) && ...
                        (next_departure_minute - current_arrival_minute) >= 45
                    % 将该航班加入航线
                    current_chain1_2_3_4 = [current_chain3; FlightMatrix(m, :)];
                    current_chain_length1_2_3_4 = current_chain_length3 + 1;
                else
                        FlightChains{end+1} = current_chain3;
                    continue
                end
                for p = 1:Flight_len
                    current_chain4 = current_chain1_2_3_4;
                    current_chain_length4 = current_chain_length1_2_3_4;
                    current_arrival_airport = FlightMatrix{m, 4};
                    current_arrival_minute = ArrivalMinutes(m);
                    next_departure_airport = FlightMatrix{p, 2};
                    next_departure_minute = DepartureMinutes(p);
                        
                    % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
                    if strcmp(current_arrival_airport, next_departure_airport) && ...
                            (next_departure_minute - current_arrival_minute) >= 45
                        % 将该航班加入航线
                        current_chain1_2_3_4_5 = [current_chain4; FlightMatrix(p, :)];
                        current_chain_length1_2_3_4_5 = current_chain_length4 + 1;
                    else
                            FlightChains{end+1} = current_chain4;
                        continue
                    end

                    for n = 1:Flight_len
                        current_chain5 = current_chain1_2_3_4_5;
                        current_chain_length5 = current_chain_length1_2_3_4_5;
                        current_arrival_airport = FlightMatrix{p, 4};
                        current_arrival_minute = ArrivalMinutes(p);
                        next_departure_airport = FlightMatrix{n, 2};
                        next_departure_minute = DepartureMinutes(n);
                            
                        % 确保后续航班的起飞机场与当前航班的到达机场相同，且过站时间45分钟
                        if strcmp(current_arrival_airport, next_departure_airport) && ...
                                (next_departure_minute - current_arrival_minute) >= 45
                            % 将该航班加入航线
                            current_chain6 = [current_chain1_2_3_4_5; FlightMatrix(n, :)];
                            current_chain_length6 = current_chain_length1_2_3_4_5 + 1;
                            FlightChains{end+1} = current_chain6;
                        else
                            FlightChains{end+1} = current_chain5;
                            continue
                        end
                    end
                end
            end
        end
    end
end

% 显示结果
disp(FlightChains);

% 去除重复航班链
FlightChainsStr = cellfun(@(x) strjoin(cellfun(@num2str, x, 'UniformOutput', false), ','), FlightChains, 'UniformOutput', false); % 'UniformOutput', false允许返回不同大小的输出，这样可以生成一个元胞数组
[~, uniqueIdx] = unique(FlightChainsStr);
UniqueFlightChains = FlightChains(uniqueIdx);
disp(UniqueFlightChains);

% 一日航班结果
% 初始化结果矩阵，预分配最大可能的大小以提高效率
MaxChainLength = max(cellfun('size', UniqueFlightChains, 1));
ResultMatrix_flight_1d = zeros(length(UniqueFlightChains), MaxChainLength);

% 将 UniqueFlightChains 的内容转换为矩阵
for i = 1:length(UniqueFlightChains)
    currentChain = UniqueFlightChains{i}(:, 1);
    ResultMatrix_flight_1d(i, 1:length(currentChain)) = vertcat(currentChain{:});
end

% 显示结果矩阵
disp(ResultMatrix_flight_1d);

% 第一日过夜机场结果
ResultMatrix_airport_1d = {};
for i = 1:length(UniqueFlightChains)
    currentairport = UniqueFlightChains{i}(end, 4);
    ResultMatrix_airport_1d{end+1} = currentairport;
end

% 显示矩阵
disp(ResultMatrix_airport_1d);

% 连接第二日航班
ResultMatrix_filght_1d_connections = [];
for i = 1:length(UniqueFlightChains)
    airport = ResultMatrix_airport_1d{i};
    currentRow = [];
    for j = 1:Flight_len  % 遍历所有航班
        if strcmp(FlightMatrix{j, 2}, airport)
            currentRow(end+1) = FlightMatrix{j, 1};
        end
    end
    ResultMatrix_filght_1d_connections(i, 1:length(currentRow)) = currentRow;
end

% 显示结果矩阵
disp(ResultMatrix_filght_1d_connections);

ResultMatrix_flight_2d = [];
for i = 1:length(UniqueFlightChains)
    for j = 1:size(ResultMatrix_filght_1d_connections, 2)
        if ResultMatrix_filght_1d_connections(i, j) ~= 0
            % 查找匹配的航班号的行
            matchingRows = find(ResultMatrix_flight_1d(:, 1) == ResultMatrix_filght_1d_connections(i, j));
            for k = 1:length(matchingRows)
                % 获取当前匹配行
                currentRow = ResultMatrix_flight_1d(matchingRows(k), :);
                % 将当前航班链与匹配行合并
                newRow = [ResultMatrix_flight_1d(i, :), currentRow];
                ResultMatrix_flight_2d(end + 1, 1:length(newRow)) = newRow;  % 确保行数一致
            end
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_2d);

% 第二日航班结果
ResultMatrix_filght_2d_last = zeros(size(ResultMatrix_flight_2d, 1), 1);

% 遍历 ResultMatrix2 的每一行
for i = 1:size(ResultMatrix_flight_2d, 1)
    % 获取当前行
    currentRow = ResultMatrix_flight_2d(i, :);
    % 找到最后一个不为0的元素
    lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
    % 存储最后一个不为0的元素
    ResultMatrix_filght_2d_last(i) = currentRow(lastNonZeroIdx);
end

% 显示结果矩阵
disp(ResultMatrix_filght_2d_last);

% 第二日最后机场结果
ResultMatrix_airport_2d = {};
for i = 1:length(ResultMatrix_filght_2d_last)
    for j = 1:size(FlightMatrix, 1)
        if ResultMatrix_filght_2d_last(i) == FlightMatrix{j, 1}
            ResultMatrix_airport_2d{end+1} = FlightMatrix{j, 4};
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_2d);

% 连接第三日航班
ResultMatrix_flight_2d_connections = [];
for i = 1:length(ResultMatrix_flight_2d)
    airport2 = ResultMatrix_airport_2d{i};
    currentRow = [];
    for j = 1:Flight_len  % 遍历所有航班
        if strcmp(FlightMatrix{j, 2}, airport2)
            currentRow(end+1) = FlightMatrix{j, 1};
        end
    end
    ResultMatrix_flight_2d_connections(i, 1:length(currentRow)) = currentRow;
end

% 显示结果矩阵
disp(ResultMatrix_flight_2d_connections);

ResultMatrix_flight_3d = [];
for i = 1:length(ResultMatrix_flight_2d)
    for j = 1:size(ResultMatrix_flight_2d_connections, 2)
        if ResultMatrix_flight_2d_connections(i, j) ~= 0
            % 查找匹配的航班号的行
            matchingRows = find(ResultMatrix_flight_1d(:, 1) == ResultMatrix_flight_2d_connections(i, j));
            for k = 1:length(matchingRows)
                % 获取当前匹配行
                currentRow = ResultMatrix_flight_1d(matchingRows(k), :);
                % 将当前航班链与匹配行合并
                newRow = [ResultMatrix_flight_2d(i, :), currentRow];
                ResultMatrix_flight_3d(end + 1, 1:length(newRow)) = newRow;  % 确保行数一致
            end
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_3d);

maxnum_1d = size(ResultMatrix_flight_3d, 2)/3;
ResultMatrix_flight_3d_last3 = [];
for i = 1:3
    for j = 1:size(ResultMatrix_flight_3d, 1)
        % 获取当前行
        currentRow = ResultMatrix_flight_3d(j, maxnum_1d * (i - 1) + 1:maxnum_1d * i);
        % 找到最后一个不为0的元素
        lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
        % 存储最后一个不为0的元素
        ResultMatrix_flight_3d_last3(j, i) = currentRow(lastNonZeroIdx);
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_3d_last3);

% JFK机场停场过夜条件
% 不同的FlightMatrix，需要改为不同的所有在JFK机场过夜的航班集合
ResultMatrix_flight_3d_final1 = ResultMatrix_flight_3d;

% 找到第四列为'JFK'的行
rows = strcmp(FlightMatrix(:, 4), 'JFK');

% 提取航班号
excludedFlights = cell2mat(FlightMatrix(rows, 1));

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = true(size(ResultMatrix_flight_3d_final1, 1), 1);

for i = 1:size(ResultMatrix_flight_3d_final1, 1)
    n = 0;  % 计数器
    for j = 1:3
        if ~ismember(ResultMatrix_flight_3d_last3(i, j), excludedFlights)  % 检查是否在排除列表中
            n = n + 1;
        end
    end
    if n == 3
        rowsToKeep(i) = false;  % 标记该行需要被删除
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_3d_final1 = ResultMatrix_flight_3d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_3d_final1);

% 起始终止机场条件
ResultMatrix_flight_3d_final1_last3 = [];

for i = 1:3
    for j = 1:size(ResultMatrix_flight_3d_final1, 1)
        % 获取当前行
        currentRow = ResultMatrix_flight_3d_final1(j, maxnum_1d * (i - 1) + 1:maxnum_1d * i);
        % 找到最后一个不为0的元素
        lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
        % 存储最后一个不为0的元素
        ResultMatrix_flight_3d_final1_last3(j, i) = currentRow(lastNonZeroIdx);
    end
end

ResultMatrix_flight_1d_first_last = [ResultMatrix_flight_3d_final1(:, 1), ResultMatrix_flight_3d_final1_last3(:, 3)];
ResultMatrix_airport_3d_first_last = cell(size(ResultMatrix_flight_1d_first_last, 1), 2);

for i = 1:size(ResultMatrix_flight_1d_first_last, 1)
    % 查找起始机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 1) == FlightMatrix{j, 1}
            ResultMatrix_airport_3d_first_last{i, 1} = FlightMatrix{j, 2};  % 起飞机场
            break;  % 找到后跳出循环
        end
    end
    
    % 查找终止机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 2) == FlightMatrix{j, 1}
            ResultMatrix_airport_3d_first_last{i, 2} = FlightMatrix{j, 4};  % 到达机场
            break;  % 找到后跳出循环
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_3d_first_last);

ResultMatrix_flight_1d_final1 = ResultMatrix_flight_3d_final1;

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = true(size(ResultMatrix_flight_1d_final1, 1), 1);

for i = 1:size(ResultMatrix_flight_1d_final1, 1)
    if ~strcmp(ResultMatrix_airport_3d_first_last{i, 1}, ResultMatrix_airport_3d_first_last{i, 2})
        rowsToKeep(i) = false;  % 标记该行需要被删除
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_1d_final1 = ResultMatrix_flight_1d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_1d_final1);

% 添加维护机会列
ResultMatrix_flight_3d_final_last3 = [];
for i = 1:3
    for j = 1:size(ResultMatrix_flight_1d_final1, 1)
        % 获取当前行
        currentRow = ResultMatrix_flight_1d_final1(j, maxnum_1d * (i - 1) + 1:maxnum_1d * i);
        % 找到最后一个不为0的元素
        lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
        % 存储最后一个不为0的元素
        ResultMatrix_flight_3d_final_last3(j, i) = currentRow(lastNonZeroIdx);
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_3d_final_last3);

MaintenanceOpp = [];
for i = 1:size(ResultMatrix_flight_3d_final_last3, 1)
    n = 0;
    for j = 1:3
        if ismember(ResultMatrix_flight_3d_final_last3(i, j), excludedFlights)
            n = n + 1;
        end
    end
    MaintenanceOpp(i, 1) = n;
end

% 显示结果矩阵
disp(MaintenanceOpp);

ResultMatrix = [ResultMatrix_flight_1d_final1, MaintenanceOpp];

% 显示结果矩阵
disp(ResultMatrix);

% 优化
VarNum = size(ResultMatrix, 1);
Aeq_size = 3 * Flight_len;

% 航班覆盖
FlightCover = {};
for i = 1:Flight_len
    CurrentRow = []; % 每次循环重置
    for j = 1:3
        n = 0;
        for k = 1:VarNum
            if ismember(FlightMatrix{i, 1}, ResultMatrix(k, maxnum_1d * (j - 1) + 1:maxnum_1d * j))
                n = n + 1;
                CurrentRow(j, n) = k;
            end
        end
    end
    FlightCover = [FlightCover; {CurrentRow}];
end

% 显示结果矩阵
disp(FlightCover);

f = -ResultMatrix(:, end)';
intcon = 1:1:VarNum;
A = ones(1, VarNum);
B = 12; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%根据需要修改
Aeq = zeros(Aeq_size, VarNum);
Beq = ones(Aeq_size, 1);
lb = zeros(1, VarNum);
ub = ones(1, VarNum);

n = 0;
for i = 1:Flight_len
    for j = 1:3
        n = n + 1;
        
        % 确保 FlightCover{i}(j, :) 是一个有效的索引数组
        indices = FlightCover{i}(j, :);
        indices = indices(indices > 0); % 排除非正整数的情况
        
        % 如果 indices 不为空，则将 Aeq 中这些位置设为 1
        if ~isempty(indices)
            Aeq(n, indices) = 1;
        end
    end
end

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x;
z = -z

%%
% 9.机组排班

% B757-200
clc; clear;
FlightMatrix = {
    125, 'JFK', '07:25', 'SFO', '09:55', 5.5;
    110, 'ATL', '08:10', 'JFK', '10:40', 2.5;
    113, 'MIA', '09:10', 'JFK', '12:10', 3;
    131, 'JFK', '09:30', 'ATL', '12:00', 2.5;
    105, 'SFO', '09:50', 'JFK', '18:20', 5.5;
    138, 'JFK', '12:30', 'BOS', '14:00', 1.5;
    111, 'ATL', '13:10', 'JFK', '15:40', 2.5;
    114, 'MIA', '14:30', 'JFK', '17:30', 3;
    118, 'BOS', '15:00', 'JFK', '16:30', 1.5;
    135, 'JFK', '15:10', 'MIA', '18:10', 3;
    133, 'JFK', '18:05', 'ATL', '20:35', 2.5;
    136, 'JFK', '18:10', 'MIA', '21:10', 3
};

Flight_len = size(FlightMatrix, 1);

% 获取起飞时间和到达时间列
DepartureTimes = FlightMatrix(:, 3);
ArrivalTimes = FlightMatrix(:, 5);

% 初始化分钟数组
DepartureMinutes = zeros(size(DepartureTimes));
ArrivalMinutes = zeros(size(ArrivalTimes));

% 转换时间为分钟数
for i = 1:length(DepartureTimes)
    % 转换起飞时间
    depTime = datetime(DepartureTimes{i}, 'InputFormat', 'HH:mm');
    DepartureMinutes(i) = hour(depTime) * 60 + minute(depTime);
    
    % 转换到达时间
    arrTime = datetime(ArrivalTimes{i}, 'InputFormat', 'HH:mm');
    ArrivalMinutes(i) = hour(arrTime) * 60 + minute(arrTime);
end

FlightChains = {};  % 用于存储航线

% 查找航线
for i = 1:Flight_len
    
    for j = 1:Flight_len
        current_chain1 = FlightMatrix(i, :);
        current_chain_length1 = 1;
        current_arrival_airport = FlightMatrix{i, 4};
        current_arrival_minute = ArrivalMinutes(i);
        next_departure_airport = FlightMatrix{j, 2};
        next_departure_minute = DepartureMinutes(j);
        
        % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
        if strcmp(current_arrival_airport, next_departure_airport) && ...
           (next_departure_minute - current_arrival_minute) >= 10 && ...
           (next_departure_minute - current_arrival_minute) <= 180
            % 将该航班加入航线
            current_chain1_2 = [current_chain1; FlightMatrix(j, :)];
            current_chain_length1_2 = current_chain_length1 + 1;
        else
            FlightChains{end+1} = current_chain1;
            continue
        end

        for k = 1:Flight_len
            current_chain2 = current_chain1_2;
            current_chain_length2 = current_chain_length1_2;
            current_arrival_airport = FlightMatrix{j, 4};
            current_arrival_minute = ArrivalMinutes(j);
            next_departure_airport = FlightMatrix{k, 2};
            next_departure_minute = DepartureMinutes(k);
                
            % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
            if strcmp(current_arrival_airport, next_departure_airport) && ...
               (next_departure_minute - current_arrival_minute) >= 10 && ...
               (next_departure_minute - current_arrival_minute) <= 180
                % 将该航班加入航线
                current_chain1_2_3 = [current_chain2; FlightMatrix(k, :)];
                current_chain_length1_2_3 = current_chain_length2 + 1;
            else
                % 保存到 FlightChains
                FlightChains{end+1} = current_chain2;
                continue
            end

            for m = 1:Flight_len
                current_chain3 = current_chain1_2_3;
                current_chain_length3 = current_chain_length1_2_3;
                current_arrival_airport = FlightMatrix{k, 4};
                current_arrival_minute = ArrivalMinutes(k);
                next_departure_airport = FlightMatrix{m, 2};
                next_departure_minute = DepartureMinutes(m);
                    
                % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
                if strcmp(current_arrival_airport, next_departure_airport) && ...
                   (next_departure_minute - current_arrival_minute) >= 10 && ...
                   (next_departure_minute - current_arrival_minute) <= 180
                    % 将该航班加入航线
                    current_chain1_2_3_4 = [current_chain3; FlightMatrix(m, :)];
                    current_chain_length1_2_3_4 = current_chain_length3 + 1;
                else
                        FlightChains{end+1} = current_chain3;
                    continue
                end
                for p = 1:Flight_len
                    current_chain4 = current_chain1_2_3_4;
                    current_chain_length4 = current_chain_length1_2_3_4;
                    current_arrival_airport = FlightMatrix{m, 4};
                    current_arrival_minute = ArrivalMinutes(m);
                    next_departure_airport = FlightMatrix{p, 2};
                    next_departure_minute = DepartureMinutes(p);
                        
                    % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
                    if strcmp(current_arrival_airport, next_departure_airport) && ...
                       (next_departure_minute - current_arrival_minute) >= 10 && ...
                       (next_departure_minute - current_arrival_minute) <= 180
                        % 将该航班加入航线
                        current_chain1_2_3_4_5 = [current_chain4; FlightMatrix(p, :)];
                        current_chain_length1_2_3_4_5 = current_chain_length4 + 1;
                    else
                            FlightChains{end+1} = current_chain4;
                        continue
                    end

                    for n = 1:Flight_len
                        current_chain5 = current_chain1_2_3_4_5;
                        current_chain_length5 = current_chain_length1_2_3_4_5;
                        current_arrival_airport = FlightMatrix{p, 4};
                        current_arrival_minute = ArrivalMinutes(p);
                        next_departure_airport = FlightMatrix{n, 2};
                        next_departure_minute = DepartureMinutes(n);
                            
                        % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
                        if strcmp(current_arrival_airport, next_departure_airport) && ...
                           (next_departure_minute - current_arrival_minute) >= 10 && ...
                           (next_departure_minute - current_arrival_minute) <= 180
                            % 将该航班加入航线
                            current_chain6 = [current_chain1_2_3_4_5; FlightMatrix(n, :)];
                            current_chain_length6 = current_chain_length1_2_3_4_5 + 1;
                            FlightChains{end+1} = current_chain6;
                        else
                            FlightChains{end+1} = current_chain5;
                            continue
                        end
                    end
                end
            end
        end
    end
end

% 显示结果
disp(FlightChains);

% 去除重复航班链
FlightChainsStr = cellfun(@(x) strjoin(cellfun(@num2str, x, 'UniformOutput', false), ','), FlightChains, 'UniformOutput', false); % 'UniformOutput', false允许返回不同大小的输出，这样可以生成一个元胞数组
[~, uniqueIdx] = unique(FlightChainsStr);
UniqueFlightChains = FlightChains(uniqueIdx);
disp(UniqueFlightChains);

% 一日航班结果
% 初始化结果矩阵，预分配最大可能的大小以提高效率
MaxChainLength = max(cellfun('size', UniqueFlightChains, 1));
ResultMatrix_flight_1d = zeros(length(UniqueFlightChains), MaxChainLength);

% 将 UniqueFlightChains 的内容转换为矩阵
for i = 1:length(UniqueFlightChains)
    currentChain = UniqueFlightChains{i}(:, 1);
    ResultMatrix_flight_1d(i, 1:length(currentChain)) = vertcat(currentChain{:});
end

% 显示结果矩阵
disp(ResultMatrix_flight_1d);

% 一日周期
% 起始和结束机场同为常驻基地JFK条件
ResultMatrix_filght_1d_last = zeros(size(ResultMatrix_flight_1d, 1), 1);

% 遍历每一行
for i = 1:size(ResultMatrix_flight_1d, 1)
    % 获取当前行
    currentRow = ResultMatrix_flight_1d(i, :);
    % 找到最后一个不为0的元素
    lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
    % 存储最后一个不为0的元素
    ResultMatrix_filght_1d_last(i) = currentRow(lastNonZeroIdx);
end

% 显示结果
disp(ResultMatrix_filght_1d_last);

ResultMatrix_flight_1d_first_last = [ResultMatrix_flight_1d(:, 1), ResultMatrix_filght_1d_last];
ResultMatrix_airport_1d_first_last = cell(size(ResultMatrix_flight_1d_first_last, 1), 2);

for i = 1:size(ResultMatrix_flight_1d_first_last, 1)
    % 查找起始机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 1) == FlightMatrix{j, 1}
            ResultMatrix_airport_1d_first_last{i, 1} = FlightMatrix{j, 2};  % 起飞机场
            break;  % 找到后跳出循环
        end
    end
    
    % 查找终止机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 2) == FlightMatrix{j, 1}
            ResultMatrix_airport_1d_first_last{i, 2} = FlightMatrix{j, 4};  % 到达机场
            break;  % 找到后跳出循环
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_1d_first_last);

ResultMatrix_flight_1d_final1 = ResultMatrix_flight_1d;

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = false(size(ResultMatrix_flight_1d_final1, 1), 1);

% 遍历每一行，检查起点和终点是否都为 'JFK'
for i = 1:size(ResultMatrix_flight_1d_final1, 1)
    if strcmp(ResultMatrix_airport_1d_first_last{i, 1}, 'JFK') && strcmp(ResultMatrix_airport_1d_first_last{i, 2}, 'JFK')
        rowsToKeep(i) = true;  % 标记该行需要保留
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_1d_final1 = ResultMatrix_flight_1d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_1d_final1);

maxnum_1d = size(ResultMatrix_flight_1d, 2);

% 每天飞行出勤时间（按飞行时间算）不能超过8小时条件
FlyTime_1d = [];
for i = 1:size(ResultMatrix_flight_1d_final1, 1)
    currenttime = 0;
    for j = 1:size(ResultMatrix_flight_1d_final1, 2)
        for k = 1:Flight_len
            if ResultMatrix_flight_1d_final1(i, j) == FlightMatrix{k, 1}
                currenttime = currenttime + FlightMatrix{k, 6};
            end
        end
    end
    FlyTime_1d(i, 1) = currenttime;
end

ResultMatrix_flight_1d_final2 = [ResultMatrix_flight_1d_final1, FlyTime_1d];

% 显示结果矩阵
disp(ResultMatrix_flight_1d_final2);
        
ResultMatrix_1d = ResultMatrix_flight_1d_final2(ResultMatrix_flight_1d_final2(:, end) <= 8, :) ;

% 显示结果矩阵
disp(ResultMatrix_1d);

% 第一日过夜机场结果
ResultMatrix_airport_1d = {};
for i = 1:length(UniqueFlightChains)
    currentairport = UniqueFlightChains{i}(end, 4);
    ResultMatrix_airport_1d{end+1} = currentairport;
end

% 显示第一日过夜机场矩阵
disp(ResultMatrix_airport_1d);

% 连接第二日航班
ResultMatrix_filght_1d_connections = [];
for i = 1:length(UniqueFlightChains)
    airport = ResultMatrix_airport_1d{i};
    currentRow = [];
    for j = 1:Flight_len  % 遍历所有航班
        if strcmp(FlightMatrix{j, 2}, airport)
            currentRow(end+1) = FlightMatrix{j, 1};  % 添加航班号到当前行
        end
    end
    ResultMatrix_filght_1d_connections(i, 1:length(currentRow)) = currentRow;
end

% 显示结果矩阵
disp(ResultMatrix_filght_1d_connections);

ResultMatrix_flight_2d = [];
for i = 1:length(UniqueFlightChains)
    for j = 1:size(ResultMatrix_filght_1d_connections, 2)
        if ResultMatrix_filght_1d_connections(i, j) ~= 0
            % 查找匹配的航班号的行
            matchingRows = find(ResultMatrix_flight_1d(:, 1) == ResultMatrix_filght_1d_connections(i, j));
            for k = 1:length(matchingRows)
                % 获取当前匹配行
                currentRow = ResultMatrix_flight_1d(matchingRows(k), :);
                % 将当前航班链与匹配行合并
                newRow = [ResultMatrix_flight_1d(i, :), currentRow];
                ResultMatrix_flight_2d(end + 1, 1:length(newRow)) = newRow;  % 确保行数一致
            end
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_2d);

% 两日周期
% 起始和结束机场同为常驻基地JFK条件
ResultMatrix_filght_2d_last = zeros(size(ResultMatrix_flight_2d, 1), 1);

% 遍历每一行
for i = 1:size(ResultMatrix_flight_2d, 1)
    % 获取当前行
    currentRow = ResultMatrix_flight_2d(i, :);
    % 找到最后一个不为0的元素
    lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
    % 存储最后一个不为0的元素
    ResultMatrix_filght_2d_last(i) = currentRow(lastNonZeroIdx);
end

% 显示结果
disp(ResultMatrix_filght_2d_last);

ResultMatrix_flight_2d_first_last = [ResultMatrix_flight_2d(:, 1), ResultMatrix_filght_2d_last];
ResultMatrix_airport_2d_first_last = cell(size(ResultMatrix_flight_2d_first_last, 1), 2);

for i = 1:size(ResultMatrix_flight_2d_first_last, 1)
    % 查找起始机场
    for j = 1:Flight_len
        if ResultMatrix_flight_2d_first_last(i, 1) == FlightMatrix{j, 1}
            ResultMatrix_airport_2d_first_last{i, 1} = FlightMatrix{j, 2};  % 起飞机场
            break;  % 找到后跳出循环
        end
    end
    
    % 查找终止机场
    for j = 1:Flight_len
        if ResultMatrix_flight_2d_first_last(i, 2) == FlightMatrix{j, 1}
            ResultMatrix_airport_2d_first_last{i, 2} = FlightMatrix{j, 4};  % 到达机场
            break;  % 找到后跳出循环
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_2d_first_last);

ResultMatrix_flight_2d_final1 = ResultMatrix_flight_2d;

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = false(size(ResultMatrix_flight_2d_final1, 1), 1);

% 遍历每一行，检查起点和终点是否都为 'JFK'
for i = 1:size(ResultMatrix_flight_2d_final1, 1)
    if strcmp(ResultMatrix_airport_2d_first_last{i, 1}, 'JFK') && strcmp(ResultMatrix_airport_2d_first_last{i, 2}, 'JFK')
        rowsToKeep(i) = true;  % 标记该行需要保留
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_2d_final1 = ResultMatrix_flight_2d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_2d_final1);

% 每天飞行出勤时间（按飞行时间算）不能超过8小时条件
FlyTime_2d = zeros(size(ResultMatrix_flight_2d_final1, 1), 2);

% 遍历每一行
for i = 1:size(ResultMatrix_flight_2d_final1, 1)
    for j = 1:2
        currenttime = 0;
        currentrow = ResultMatrix_flight_2d_final1(i, maxnum_1d * (j - 1) + 1:maxnum_1d * j);
        % 遍历每个航班编号
        for k = 1:maxnum_1d
            % 遍历 FlightMatrix 来查找对应的航班信息
            for m = 1:Flight_len
                if currentrow(k) == FlightMatrix{m, 1}
                    currenttime = currenttime + FlightMatrix{m, 6};
                end
            end
        end
    FlyTime_2d(i, j) = currenttime;
    end
end

% 显示结果
disp(FlyTime_2d);

ResultMatrix_flight_2d_final2 = [ResultMatrix_flight_2d_final1, FlyTime_2d];

% 显示结果矩阵
disp(ResultMatrix_flight_2d_final2);
        
ResultMatrix_2d = ResultMatrix_flight_2d_final2(ResultMatrix_flight_2d_final2(:, end-1) <= 8 & ResultMatrix_flight_2d_final2(:, end) <= 8, :);

% 显示结果矩阵
disp(ResultMatrix_2d);

ResultMatrix_1dto2d = [ResultMatrix_1d(:, 1:end-1), zeros(size(ResultMatrix_1d, 1), size(ResultMatrix_1d, 2)-1)];
ResultMatrix = [ResultMatrix_1dto2d; ResultMatrix_2d(:, 1:end-2)];

% 优化
VarNum = size(ResultMatrix, 1);
Aeq_size = Flight_len;

% 航班覆盖
FlightCover = {};
for i = 1:Flight_len
    CurrentRow = []; % 每次循环重置
    n = 0;
    for j = 1:VarNum
        if ismember(FlightMatrix{i, 1}, ResultMatrix(j, :))
            n = n + 1;
            CurrentRow(1, n) = j;
        end
    end
    FlightCover = [FlightCover; {CurrentRow}];
end

% 显示结果矩阵
disp(FlightCover);

f1 = ones(1, size(ResultMatrix_1d, 1));
f2 = 3 * ones(1, size(ResultMatrix_2d, 1));
f = [f1, f2];
intcon = 1:1:VarNum;
A = [];
B = []; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%根据需要修改
Aeq = zeros(Aeq_size, VarNum);
Beq = ones(Aeq_size, 1);
lb = zeros(1, VarNum);
ub = ones(1, VarNum);

for i = 1:Flight_len

    % 确保 FlightCover{i}(j, :) 是一个有效的索引数组
    indices = FlightCover{i};
    indices = indices(indices > 0); % 排除非正整数的情况
        
    % 如果 indices 不为空，则将 Aeq 中这些位置设为 1
    if ~isempty(indices)
        Aeq(i, indices) = 1;
    end
end

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

peidui = find(x ~= 0);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

% 机组轮班
FlyTime_1dto2d = [FlyTime_1d, zeros(size(FlyTime_1d, 1), size(FlyTime_1d, 2))];
ResultMatrix_final = [ResultMatrix_1dto2d, FlyTime_1dto2d; ResultMatrix_2d];
FlyTime = ResultMatrix_final(:, end-1) + ResultMatrix_final(:, end);
PeiDui_flytime = FlyTime(peidui);
A = [];
PeiDuiNum = length(peidui);

% 定义i和j的数组
i_vals = [1, 1, 2, 2, 3, 3, 4];
j_vals = [4, 5, 5, 6, 6, 7, 7];

for k = 1:length(i_vals)
    i = i_vals(k);
    j = j_vals(k);

    for m = 1:PeiDuiNum
        B = zeros(PeiDuiNum, 10);
        B(1:PeiDuiNum, i) = m;
        B(1:PeiDuiNum, 8) = PeiDui_flytime(m);
        for n = 1:PeiDuiNum
            B(n, j) = n;
            B(n, 9) = PeiDui_flytime(n);
        end
        B(:, 10) = B(:, 8) + B(:, 9);
        A = [A; B];
    end
end

Result = [A(:, 1:7), A(:, 10)];

% 显示结果矩阵
disp(Result);

% 优化
Var_Num = 7 * PeiDuiNum^2; 
f_LunBan = abs(Result(:, end) - 20)';
intcon_LunBan = 1:1:Var_Num;
A_LunBan = [];
B_LunBan = [];
Aeq_LunBan = zeros(7 * PeiDuiNum, Var_Num);
Beq_LunBan = ones(7 * PeiDuiNum, 1);
lb_LunBan = zeros(1, Var_Num);
ub_LunBan = ones(1, Var_Num);

Aeq_LunBan_1 = [];

n = 0;
for i = 1:7
    for j = 1:PeiDuiNum
        n = n + 1;
        newrow = find(Result(:, i) == j)';
        Aeq_LunBan_1(n, 1:length(newrow)) = newrow;
    end
end

for i = 1:size(Aeq_LunBan_1, 1)
    Aeq_LunBan(i, Aeq_LunBan_1(i, :)) = 1;
end

x_LunBan = intlinprog(f_LunBan, intcon_LunBan, A_LunBan, B_LunBan, Aeq_LunBan, Beq_LunBan, lb_LunBan, ub_LunBan);

ResultIndexes= find(x_LunBan ~= 0);

ResultValues = x_LunBan(find(x_LunBan ~= 0));

Result = [ResultIndexes, ResultValues]

z_LunBan = f_LunBan * x_LunBan



% B737-800
clc; clear;
FlightMatrix = {
    101, 'LAX', '05:00', 'JFK', '13:30', 5.5;
    104, 'SFO', '05:05', 'JFK', '13:35', 5.5;
    116, 'BOS', '06:15', 'JFK', '07:45', 1.5;
    140, 'JFK', '06:20', 'IAD', '07:20', 1;
    107, 'ORD', '07:30', 'JFK', '10:30', 2;
    122, 'JFK', '07:35', 'LAX', '10:05', 5.5;
    137, 'JFK', '07:40', 'BOS', '09:10', 1.5;
    119, 'IAD', '08:15', 'JFK', '09:15', 1;
    102, 'LAX', '09:45', 'JFK', '18:15', 5.5;
    117, 'BOS', '10:00', 'JFK', '11:30', 1.5;
    128, 'JFK', '10:05', 'ORD', '11:05', 2;
    134, 'JFK', '10:35', 'MIA', '13:35', 3;
    141, 'JFK', '12:00', 'IAD', '13:00', 1;
    108, 'ORD', '12:20', 'JFK', '15:20', 2;
    120, 'IAD', '14:25', 'JFK', '15:25', 1;
    132, 'JFK', '14:35', 'ATL', '17:35', 2.5;
    129, 'JFK', '15:05', 'ORD', '16:05', 2;
    142, 'JFK', '15:15', 'IAD', '16:15', 1;
    103, 'LAX', '15:20', 'JFK', '23:50', 5.5;
    106, 'SFO', '15:25', 'JFK', '23:55', 5.5;
    126, 'JFK', '15:30', 'SFO', '18:00', 5.5;
    123, 'JFK', '16:00', 'LAX', '18:30', 5.5;
    109, 'ORD', '17:10', 'JFK', '20:10', 2;
    112, 'ATL', '18:00', 'JFK', '20:30', 2.5;
    115, 'MIA', '18:15', 'JFK', '21:15', 3;
    121, 'IAD', '18:30', 'JFK', '19:30', 1;
    124, 'JFK', '19:00', 'LAX', '21:30', 5.5;
    127, 'JFK', '20:00', 'SFO', '22:30', 5.5;
    130, 'JFK', '21:00', 'ORD', '22:00', 2;
    139, 'JFK', '21:30', 'BOS', '23:00', 1.5
};

Flight_len = size(FlightMatrix, 1);

% 获取起飞时间和到达时间列
DepartureTimes = FlightMatrix(:, 3);
ArrivalTimes = FlightMatrix(:, 5);

% 初始化分钟数组
DepartureMinutes = zeros(size(DepartureTimes));
ArrivalMinutes = zeros(size(ArrivalTimes));

% 转换时间为分钟数
for i = 1:length(DepartureTimes)
    % 转换起飞时间
    depTime = datetime(DepartureTimes{i}, 'InputFormat', 'HH:mm');
    DepartureMinutes(i) = hour(depTime) * 60 + minute(depTime);
    
    % 转换到达时间
    arrTime = datetime(ArrivalTimes{i}, 'InputFormat', 'HH:mm');
    ArrivalMinutes(i) = hour(arrTime) * 60 + minute(arrTime);
end

FlightChains = {};  % 用于存储航线

% 查找航线
for i = 1:Flight_len
    
    for j = 1:Flight_len
        current_chain1 = FlightMatrix(i, :);
        current_chain_length1 = 1;
        current_arrival_airport = FlightMatrix{i, 4};
        current_arrival_minute = ArrivalMinutes(i);
        next_departure_airport = FlightMatrix{j, 2};
        next_departure_minute = DepartureMinutes(j);
        
        % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
        if strcmp(current_arrival_airport, next_departure_airport) && ...
           (next_departure_minute - current_arrival_minute) >= 10 && ...
           (next_departure_minute - current_arrival_minute) <= 180
            % 将该航班加入航线
            current_chain1_2 = [current_chain1; FlightMatrix(j, :)];
            current_chain_length1_2 = current_chain_length1 + 1;
        else
            FlightChains{end+1} = current_chain1;
            continue
        end

        for k = 1:Flight_len
            current_chain2 = current_chain1_2;
            current_chain_length2 = current_chain_length1_2;
            current_arrival_airport = FlightMatrix{j, 4};
            current_arrival_minute = ArrivalMinutes(j);
            next_departure_airport = FlightMatrix{k, 2};
            next_departure_minute = DepartureMinutes(k);
                
            % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
            if strcmp(current_arrival_airport, next_departure_airport) && ...
               (next_departure_minute - current_arrival_minute) >= 10 && ...
               (next_departure_minute - current_arrival_minute) <= 180
                % 将该航班加入航线
                current_chain1_2_3 = [current_chain2; FlightMatrix(k, :)];
                current_chain_length1_2_3 = current_chain_length2 + 1;
            else
                % 保存到 FlightChains
                FlightChains{end+1} = current_chain2;
                continue
            end

            for m = 1:Flight_len
                current_chain3 = current_chain1_2_3;
                current_chain_length3 = current_chain_length1_2_3;
                current_arrival_airport = FlightMatrix{k, 4};
                current_arrival_minute = ArrivalMinutes(k);
                next_departure_airport = FlightMatrix{m, 2};
                next_departure_minute = DepartureMinutes(m);
                    
                % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
                if strcmp(current_arrival_airport, next_departure_airport) && ...
                   (next_departure_minute - current_arrival_minute) >= 10 && ...
                   (next_departure_minute - current_arrival_minute) <= 180
                    % 将该航班加入航线
                    current_chain1_2_3_4 = [current_chain3; FlightMatrix(m, :)];
                    current_chain_length1_2_3_4 = current_chain_length3 + 1;
                else
                        FlightChains{end+1} = current_chain3;
                    continue
                end
                for p = 1:Flight_len
                    current_chain4 = current_chain1_2_3_4;
                    current_chain_length4 = current_chain_length1_2_3_4;
                    current_arrival_airport = FlightMatrix{m, 4};
                    current_arrival_minute = ArrivalMinutes(m);
                    next_departure_airport = FlightMatrix{p, 2};
                    next_departure_minute = DepartureMinutes(p);
                        
                    % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
                    if strcmp(current_arrival_airport, next_departure_airport) && ...
                       (next_departure_minute - current_arrival_minute) >= 10 && ...
                       (next_departure_minute - current_arrival_minute) <= 180
                        % 将该航班加入航线
                        current_chain1_2_3_4_5 = [current_chain4; FlightMatrix(p, :)];
                        current_chain_length1_2_3_4_5 = current_chain_length4 + 1;
                    else
                            FlightChains{end+1} = current_chain4;
                        continue
                    end

                    for n = 1:Flight_len
                        current_chain5 = current_chain1_2_3_4_5;
                        current_chain_length5 = current_chain_length1_2_3_4_5;
                        current_arrival_airport = FlightMatrix{p, 4};
                        current_arrival_minute = ArrivalMinutes(p);
                        next_departure_airport = FlightMatrix{n, 2};
                        next_departure_minute = DepartureMinutes(n);
                            
                        % 确保后续航班的起飞机场与当前航班的到达机场相同，且等待时间10-180分钟
                        if strcmp(current_arrival_airport, next_departure_airport) && ...
                           (next_departure_minute - current_arrival_minute) >= 10 && ...
                           (next_departure_minute - current_arrival_minute) <= 180
                            % 将该航班加入航线
                            current_chain6 = [current_chain1_2_3_4_5; FlightMatrix(n, :)];
                            current_chain_length6 = current_chain_length1_2_3_4_5 + 1;
                            FlightChains{end+1} = current_chain6;
                        else
                            FlightChains{end+1} = current_chain5;
                            continue
                        end
                    end
                end
            end
        end
    end
end

% 显示结果
disp(FlightChains);

% 去除重复航班链
FlightChainsStr = cellfun(@(x) strjoin(cellfun(@num2str, x, 'UniformOutput', false), ','), FlightChains, 'UniformOutput', false); % 'UniformOutput', false允许返回不同大小的输出，这样可以生成一个元胞数组
[~, uniqueIdx] = unique(FlightChainsStr);
UniqueFlightChains = FlightChains(uniqueIdx);
disp(UniqueFlightChains);

% 一日航班结果
% 初始化结果矩阵，预分配最大可能的大小以提高效率
MaxChainLength = max(cellfun('size', UniqueFlightChains, 1));
ResultMatrix_flight_1d = zeros(length(UniqueFlightChains), MaxChainLength);

% 将 UniqueFlightChains 的内容转换为矩阵
for i = 1:length(UniqueFlightChains)
    currentChain = UniqueFlightChains{i}(:, 1);
    ResultMatrix_flight_1d(i, 1:length(currentChain)) = vertcat(currentChain{:});
end

% 显示结果矩阵
disp(ResultMatrix_flight_1d);

% 一日周期
% 起始和结束机场同为常驻基地JFK条件
ResultMatrix_filght_1d_last = zeros(size(ResultMatrix_flight_1d, 1), 1);

% 遍历每一行
for i = 1:size(ResultMatrix_flight_1d, 1)
    % 获取当前行
    currentRow = ResultMatrix_flight_1d(i, :);
    % 找到最后一个不为0的元素
    lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
    % 存储最后一个不为0的元素
    ResultMatrix_filght_1d_last(i) = currentRow(lastNonZeroIdx);
end

% 显示结果
disp(ResultMatrix_filght_1d_last);

ResultMatrix_flight_1d_first_last = [ResultMatrix_flight_1d(:, 1), ResultMatrix_filght_1d_last];
ResultMatrix_airport_1d_first_last = cell(size(ResultMatrix_flight_1d_first_last, 1), 2);

for i = 1:size(ResultMatrix_flight_1d_first_last, 1)
    % 查找起始机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 1) == FlightMatrix{j, 1}
            ResultMatrix_airport_1d_first_last{i, 1} = FlightMatrix{j, 2};  % 起飞机场
            break;  % 找到后跳出循环
        end
    end
    
    % 查找终止机场
    for j = 1:Flight_len
        if ResultMatrix_flight_1d_first_last(i, 2) == FlightMatrix{j, 1}
            ResultMatrix_airport_1d_first_last{i, 2} = FlightMatrix{j, 4};  % 到达机场
            break;  % 找到后跳出循环
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_1d_first_last);

ResultMatrix_flight_1d_final1 = ResultMatrix_flight_1d;

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = false(size(ResultMatrix_flight_1d_final1, 1), 1);

% 遍历每一行，检查起点和终点是否都为 'JFK'
for i = 1:size(ResultMatrix_flight_1d_final1, 1)
    if strcmp(ResultMatrix_airport_1d_first_last{i, 1}, 'JFK') && strcmp(ResultMatrix_airport_1d_first_last{i, 2}, 'JFK')
        rowsToKeep(i) = true;  % 标记该行需要保留
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_1d_final1 = ResultMatrix_flight_1d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_1d_final1);

maxnum_1d = size(ResultMatrix_flight_1d, 2);

% 每天飞行出勤时间（按飞行时间算）不能超过8小时条件
FlyTime_1d = [];
for i = 1:size(ResultMatrix_flight_1d_final1, 1)
    currenttime = 0;
    for j = 1:size(ResultMatrix_flight_1d_final1, 2)
        for k = 1:Flight_len
            if ResultMatrix_flight_1d_final1(i, j) == FlightMatrix{k, 1}
                currenttime = currenttime + FlightMatrix{k, 6};
            end
        end
    end
    FlyTime_1d(i, 1) = currenttime;
end

ResultMatrix_flight_1d_final2 = [ResultMatrix_flight_1d_final1, FlyTime_1d];

% 显示结果矩阵
disp(ResultMatrix_flight_1d_final2);
        
ResultMatrix_1d = ResultMatrix_flight_1d_final2(ResultMatrix_flight_1d_final2(:, end) <= 8, :) ;

% 显示结果矩阵
disp(ResultMatrix_1d);

% 第一日过夜机场结果
ResultMatrix_airport_1d = {};
for i = 1:length(UniqueFlightChains)
    currentairport = UniqueFlightChains{i}(end, 4);
    ResultMatrix_airport_1d{end+1} = currentairport;
end

% 显示第一日过夜机场矩阵
disp(ResultMatrix_airport_1d);

% 连接第二日航班
ResultMatrix_filght_1d_connections = [];
for i = 1:length(UniqueFlightChains)
    airport = ResultMatrix_airport_1d{i};
    currentRow = [];
    for j = 1:Flight_len  % 遍历所有航班
        if strcmp(FlightMatrix{j, 2}, airport)
            currentRow(end+1) = FlightMatrix{j, 1};  % 添加航班号到当前行
        end
    end
    ResultMatrix_filght_1d_connections(i, 1:length(currentRow)) = currentRow;
end

% 显示结果矩阵
disp(ResultMatrix_filght_1d_connections);

ResultMatrix_flight_2d = [];
for i = 1:length(UniqueFlightChains)
    for j = 1:size(ResultMatrix_filght_1d_connections, 2)
        if ResultMatrix_filght_1d_connections(i, j) ~= 0
            % 查找匹配的航班号的行
            matchingRows = find(ResultMatrix_flight_1d(:, 1) == ResultMatrix_filght_1d_connections(i, j));
            for k = 1:length(matchingRows)
                % 获取当前匹配行
                currentRow = ResultMatrix_flight_1d(matchingRows(k), :);
                % 将当前航班链与匹配行合并
                newRow = [ResultMatrix_flight_1d(i, :), currentRow];
                ResultMatrix_flight_2d(end + 1, 1:length(newRow)) = newRow;  % 确保行数一致
            end
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_flight_2d);

% 两日周期
% 起始和结束机场同为常驻基地JFK条件
ResultMatrix_filght_2d_last = zeros(size(ResultMatrix_flight_2d, 1), 1);

% 遍历每一行
for i = 1:size(ResultMatrix_flight_2d, 1)
    % 获取当前行
    currentRow = ResultMatrix_flight_2d(i, :);
    % 找到最后一个不为0的元素
    lastNonZeroIdx = find(currentRow ~= 0, 1, 'last');
    % 存储最后一个不为0的元素
    ResultMatrix_filght_2d_last(i) = currentRow(lastNonZeroIdx);
end

% 显示结果
disp(ResultMatrix_filght_2d_last);

ResultMatrix_flight_2d_first_last = [ResultMatrix_flight_2d(:, 1), ResultMatrix_filght_2d_last];
ResultMatrix_airport_2d_first_last = cell(size(ResultMatrix_flight_2d_first_last, 1), 2);

for i = 1:size(ResultMatrix_flight_2d_first_last, 1)
    % 查找起始机场
    for j = 1:Flight_len
        if ResultMatrix_flight_2d_first_last(i, 1) == FlightMatrix{j, 1}
            ResultMatrix_airport_2d_first_last{i, 1} = FlightMatrix{j, 2};  % 起飞机场
            break;  % 找到后跳出循环
        end
    end
    
    % 查找终止机场
    for j = 1:Flight_len
        if ResultMatrix_flight_2d_first_last(i, 2) == FlightMatrix{j, 1}
            ResultMatrix_airport_2d_first_last{i, 2} = FlightMatrix{j, 4};  % 到达机场
            break;  % 找到后跳出循环
        end
    end
end

% 显示结果矩阵
disp(ResultMatrix_airport_2d_first_last);

ResultMatrix_flight_2d_final1 = ResultMatrix_flight_2d;

% 创建一个逻辑数组来标记需要保留的行
rowsToKeep = false(size(ResultMatrix_flight_2d_final1, 1), 1);

% 遍历每一行，检查起点和终点是否都为 'JFK'
for i = 1:size(ResultMatrix_flight_2d_final1, 1)
    if strcmp(ResultMatrix_airport_2d_first_last{i, 1}, 'JFK') && strcmp(ResultMatrix_airport_2d_first_last{i, 2}, 'JFK')
        rowsToKeep(i) = true;  % 标记该行需要保留
    end
end

% 根据逻辑数组过滤结果矩阵
ResultMatrix_flight_2d_final1 = ResultMatrix_flight_2d_final1(rowsToKeep, :);

% 显示结果矩阵
disp(ResultMatrix_flight_2d_final1);

% 每天飞行出勤时间（按飞行时间算）不能超过8小时条件
FlyTime_2d = zeros(size(ResultMatrix_flight_2d_final1, 1), 2);

% 遍历每一行
for i = 1:size(ResultMatrix_flight_2d_final1, 1)
    for j = 1:2
        currenttime = 0;
        currentrow = ResultMatrix_flight_2d_final1(i, maxnum_1d * (j - 1) + 1:maxnum_1d * j);
        % 遍历每个航班编号
        for k = 1:maxnum_1d
            % 遍历 FlightMatrix 来查找对应的航班信息
            for m = 1:Flight_len
                if currentrow(k) == FlightMatrix{m, 1}
                    currenttime = currenttime + FlightMatrix{m, 6};
                end
            end
        end
    FlyTime_2d(i, j) = currenttime;
    end
end

% 显示结果
disp(FlyTime_2d);

ResultMatrix_flight_2d_final2 = [ResultMatrix_flight_2d_final1, FlyTime_2d];

% 显示结果矩阵
disp(ResultMatrix_flight_2d_final2);
        
ResultMatrix_2d = ResultMatrix_flight_2d_final2(ResultMatrix_flight_2d_final2(:, end-1) <= 8 & ResultMatrix_flight_2d_final2(:, end) <= 8, :);

% 显示结果矩阵
disp(ResultMatrix_2d);

ResultMatrix_1dto2d = [ResultMatrix_1d(:, 1:end-1), zeros(size(ResultMatrix_1d, 1), size(ResultMatrix_1d, 2)-1)];
ResultMatrix = [ResultMatrix_1dto2d; ResultMatrix_2d(:, 1:end-2)];

% 优化
VarNum = size(ResultMatrix, 1);
Aeq_size = Flight_len;

% 航班覆盖
FlightCover = {};
for i = 1:Flight_len
    CurrentRow = []; % 每次循环重置
    n = 0;
    for j = 1:VarNum
        if ismember(FlightMatrix{i, 1}, ResultMatrix(j, :))
            n = n + 1;
            CurrentRow(1, n) = j;
        end
    end
    FlightCover = [FlightCover; {CurrentRow}];
end

% 显示结果矩阵
disp(FlightCover);

f1 = ones(1, size(ResultMatrix_1d, 1));
f2 = 3 * ones(1, size(ResultMatrix_2d, 1));
f = [f1, f2];
intcon = 1:1:VarNum;
A = [];
B = []; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%根据需要修改
Aeq = zeros(Aeq_size, VarNum);
Beq = ones(Aeq_size, 1);
lb = zeros(1, VarNum);
ub = ones(1, VarNum);

for i = 1:Flight_len

    % 确保 FlightCover{i}(j, :) 是一个有效的索引数组
    indices = FlightCover{i};
    indices = indices(indices > 0); % 排除非正整数的情况
        
    % 如果 indices 不为空，则将 Aeq 中这些位置设为 1
    if ~isempty(indices)
        Aeq(i, indices) = 1;
    end
end


x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

peidui = find(x ~= 0);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

% 机组轮班
FlyTime_1dto2d = [FlyTime_1d, zeros(size(FlyTime_1d, 1), size(FlyTime_1d, 2))];
ResultMatrix_final = [ResultMatrix_1dto2d, FlyTime_1dto2d; ResultMatrix_2d];
FlyTime = ResultMatrix_final(:, end-1) + ResultMatrix_final(:, end);
PeiDui_flytime = FlyTime(peidui);
Auxi = [];
PeiDuiNum = length(peidui);

% 定义i和j的数组
i_vals = [1, 1, 2, 2, 3, 3, 4];
j_vals = [4, 5, 5, 6, 6, 7, 7];

for k = 1:length(i_vals)
    i = i_vals(k);
    j = j_vals(k);

    for m = 1:PeiDuiNum
        B = zeros(PeiDuiNum, 10);
        B(1:PeiDuiNum, i) = m;
        B(1:PeiDuiNum, 8) = PeiDui_flytime(m);
        for n = 1:PeiDuiNum
            B(n, j) = n;
            B(n, 9) = PeiDui_flytime(n);
        end
        B(:, 10) = B(:, 8) + B(:, 9);
        Auxi = [Auxi; B];
    end
end

Result = [Auxi(:, 1:7), Auxi(:, 10)];

% 显示结果矩阵
disp(Result);

% 优化
Var_Num = 7 * PeiDuiNum^2; 
f_LunBan = abs(Result(:, end) - 20)';
intcon_LunBan = 1:1:Var_Num;
A_LunBan = [];
B_LunBan = [];
Aeq_LunBan = zeros(7 * PeiDuiNum, Var_Num);
Beq_LunBan = ones(7 * PeiDuiNum, 1);
lb_LunBan = zeros(1, Var_Num);
ub_LunBan = ones(1, Var_Num);

Aeq_LunBan_1 = [];

n = 0;
for i = 1:7
    for j = 1:PeiDuiNum
        n = n + 1;
        newrow = find(Result(:, i) == j)';
        Aeq_LunBan_1(n, 1:length(newrow)) = newrow;
    end
end

for i = 1:size(Aeq_LunBan_1, 1)
    Aeq_LunBan(i, Aeq_LunBan_1(i, :)) = 1;
end

x_LunBan = intlinprog(f_LunBan, intcon_LunBan, A_LunBan, B_LunBan, Aeq_LunBan, Beq_LunBan, lb_LunBan, ub_LunBan);

ResultIndexes= find(x_LunBan ~= 0);

ResultValues = x_LunBan(find(x_LunBan ~= 0));

Result = [ResultIndexes, ResultValues]

z_LunBan = f_LunBan * x_LunBan

%%
% 10.登机口分配

% 1
clc; clear;
matrix1 = [
    5 5 10 8 15 8 2 10 8 20 5 4 0 9 3 4 1 2 1;
    5 2 1 4 19 9 4 2 3 2 27 3 8 4 0 2 1 7 2;
    10 0 4 9 13 4 4 4 3 5 5 8 4 9 11 7 9 4 4;
    4 8 5 4 10 4 1 0 0 2 4 19 1 2 4 5 5 8 2;
    4 11 9 9 6 3 1 4 4 2 1 0 3 5 1 2 2 3 4;
    1 2 42 5 2 7 6 2 4 7 2 3 6 4 10 2 1 0 0;
    3 3 2 5 9 13 11 2 2 3 7 22 4 0 1 1 2 2 9
];

matrix2 = [
    10 40 0 30 10 40 20 50 30 60 40 70 50 80 60 90 70 90 80;
    40 10 30 0 40 10 50 20 60 30 70 40 80 50 90 60 90 70 80;
    70 40 60 30 50 20 40 10 30 0 40 10 50 40 60 30 70 40 50;
    50 80 40 70 30 60 20 50 10 40 0 30 10 40 20 50 30 50 40;
    90 60 80 50 70 40 60 30 50 20 40 10 30 0 40 10 50 20 30;
    70 100 60 90 50 80 40 70 30 60 20 50 10 40 0 30 10 30 20;
    80 100 70 90 60 80 50 70 40 60 30 50 20 40 10 30 0 20 10
];

TotalDistance = matrix1 * matrix2';
coef = reshape(TotalDistance', 1, []);
[m, n] = size(TotalDistance);
VarNum = 717;
STNum = 14;

f = zeros(1, VarNum);

a = 1:7;
b = [3 4 10 11 14 15 17];
f_ind = [];
for i = a
    for j = b
        f_ind = [f_ind, str2double([num2str(i) num2str(j)])];
    end
end

f(f_ind) = coef;

intcon = 1:1:VarNum;
A = [];
B = [];
Aeq = zeros(STNum, VarNum);
Beq = ones(STNum, 1);
lb = zeros(1, VarNum);
ub = ones(1, VarNum);

Aeq_1 = reshape(f_ind, 7, 7)';
Aeq_2 = reshape(f_ind, 7, 7);
Aeq_Total = [Aeq_1; Aeq_2];
for i = 1:STNum
    Aeq(i, Aeq_Total(i, :)) = 1;
end

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

% 2
clc; clear;
matrix1 = [
    5 5 10 8 15 8 2 10 8 20 5 4 0 9 3 4 1 2 1;
    5 2 1 4 19 9 4 2 3 2 27 3 8 4 0 2 1 7 2;
    10 0 4 9 13 4 4 4 3 5 5 8 4 9 11 7 9 4 4;
    4 8 5 4 10 4 1 0 0 2 4 19 1 2 4 5 5 8 2;
    4 11 9 9 6 3 1 4 4 2 1 0 3 5 1 2 2 3 4;
    1 2 42 5 2 7 6 2 4 7 2 3 6 4 10 2 1 0 0;
    3 3 2 5 9 13 11 2 2 3 7 22 4 0 1 1 2 2 9
];

matrix2 = [
    10 40 0 30 10 40 20 50 30 60 40 70 50 80 60 90 70 90 80;
    40 10 30 0 40 10 50 20 60 30 70 40 80 50 90 60 90 70 80;
    70 40 60 30 50 20 40 10 30 0 40 10 50 40 60 30 70 40 50;
    50 80 40 70 30 60 20 50 10 40 0 30 10 40 20 50 30 50 40;
    90 60 80 50 70 40 60 30 50 20 40 10 30 0 40 10 50 20 30;
    70 100 60 90 50 80 40 70 30 60 20 50 10 40 0 30 10 30 20;
    80 100 70 90 60 80 50 70 40 60 30 50 20 40 10 30 0 20 10
];

TotalDistance = matrix1 * matrix2';
coef = reshape(TotalDistance', 1, []);
[m, n] = size(TotalDistance);
VarNum = 717;
STNum = 15;

f = zeros(1, VarNum);

a = 1:7;
b = [3 4 10 11 14 15 17];
f_ind = [];
for i = a
    for j = b
        f_ind = [f_ind, str2double([num2str(i) num2str(j)])];
    end
end

f(f_ind) = coef;

intcon = 1:1:VarNum;
A = [];
B = [];
Aeq = zeros(STNum, VarNum);
Beq = ones(STNum, 1);
lb = zeros(1, VarNum);
ub = ones(1, VarNum);

Aeq_1 = reshape(f_ind, 7, 7)';
Aeq_2 = reshape(f_ind, 7, 7);
Aeq_Total = [Aeq_1; Aeq_2];
for i = 1:STNum-1
    Aeq(i, Aeq_Total(i, :)) = 1;
end
Aeq(STNum, [110, 114]) = 1;
Beq(STNum, 1) = 0;

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x

%%
% 11.人力资源规划

clc;clear;
Demand = [
    8,   8,   8,   8,  10,  10,   6;
   12,  10,  12,  10,  16,  16,   8;
   16,  12,  16,  12,  20,  20,   8;
    9,   8,   9,   8,  12,  12,   4;
];
STNum = 28;
VarNum = 73;
LunBanNum = 3;

f = ones(1, VarNum);
intcon = 1:1:VarNum;
A = zeros(STNum, VarNum);
B = -reshape(Demand', [], 1);
Aeq = [];
Beq = [];
lb = zeros(1, VarNum);
ub = [];

Auxi = [];
d7 = [1, 2, 3, 4, 5, 6, 7];
d14 = [d7, d7];

for i = 1:LunBanNum
    for j = 1:7
        newrow = [
            str2double([num2str(d14(j)), num2str(i)]), ...
            str2double([num2str(d14(j+3)), num2str(i)]), ...
            str2double([num2str(d14(j+4)), num2str(i)]), ...
            str2double([num2str(d14(j+5)), num2str(i)]), ...
            str2double([num2str(d14(j+6)), num2str(i)])
        ];
        Auxi(end+1, 1:length(newrow)) = newrow;
    end
end
LunBan1 = Auxi(1:7, :);
LunBan2 = Auxi(8:14, :);
LunBan3 = Auxi(15:21, :);
ST1 = LunBan1;
ST2 = [LunBan1, LunBan2];
ST3 = [LunBan2, LunBan3];
ST4 = Auxi(end-6:end, :);

% 确保ST1, ST2, ST3, ST4的列数一致
maxCols = max([size(ST1, 2), size(ST2, 2), size(ST3, 2), size(ST4, 2)]);

% 填充列数不足的矩阵
ST1 = [ST1, zeros(size(ST1, 1), maxCols - size(ST1, 2))];
ST2 = [ST2, zeros(size(ST2, 1), maxCols - size(ST2, 2))];
ST3 = [ST3, zeros(size(ST3, 1), maxCols - size(ST3, 2))];
ST4 = [ST4, zeros(size(ST4, 1), maxCols - size(ST4, 2))];

% 连接所有ST矩阵
ST = [ST1; ST2; ST3; ST4];

for i = 1:STNum
    ind = ST(i, :);
    A(i, ind(ind ~= 0)) = -1;
end

x = intlinprog(f, intcon, A, B, Aeq, Beq, lb, ub);

ResultIndexes= find(x ~= 0);

ResultValues = x(find(x ~= 0));

Result = [ResultIndexes, ResultValues]

z = f * x
