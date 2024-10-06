%%
%%
clc;clear;
close all;

% 导入数据
data = load('mydata');
X = data.X;

% %读取 Excel 文件
% [num_data, ~, ~] = xlsread('1999年全国31个省份城镇居民家庭平均每人全年消费性支出数据.xlsx');
% X = num_data(:,2:3); % 选择两列数据作为x轴和y轴，然后进行聚类

% 初始化参数
epsilon = 0.5; % 邻域半径
min_pts = 5;   % 邻域内最少样本数
[lbl,inie]=dbscan(X,epsilon,min_pts);

% 绘图
plotresult(X, lbl);
title(['DBSCAN 聚类结果图 (半径为：' num2str(epsilon) ', 邻域最少样本数为：' num2str(min_pts) ')']);
