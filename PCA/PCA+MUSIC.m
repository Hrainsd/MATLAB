%%
%%
clc;
clear;

m = 1000;
n = 100;

% 假设你有一个 m×n 的数据矩阵 X，其中 m 是样本数，n 是特征数
X = randn(m, n); % 这里使用随机数据，你可以替换成你的实际数据

% 计算协方差矩阵
covarianceMatrix = cov(X);

% 使用PCA获取主成分
[coeff, ~, ~, ~, explained] = pca(X);

% 计算累积贡献率
cumulativeExplained = cumsum(explained);

% 提取累积贡献率超过85%的主成分
thresholdExplained = 85;
numPrincipalComponents = find(cumulativeExplained >= thresholdExplained, 1);

% 提取主成分
principalComponents = coeff(:, 1:numPrincipalComponents);

% 显示每个主成分的贡献率和累积贡献率
disp('每个主成分的贡献率:');
disp(explained);
disp('累积贡献率:');
disp(cumulativeExplained);
disp(['累积贡献率超过', num2str(thresholdExplained), '%的主成分数量: ', num2str(numPrincipalComponents)]);

% 使用MUSIC算法估计信号方向
theta = 0:0.1:180; % 调整步长
musicSpectrum = zeros(size(theta));

for angleIdx = 1:length(theta)
    angle = theta(angleIdx);
    steeringVector = exp(1i*pi/180 * angle * (0:n-1)'); % 构建方向向量

    % MUSIC算法
    musicSpectrum(angleIdx) = 1 / (steeringVector' * pinv(principalComponents * principalComponents') * steeringVector);
end

% 规范化MUSIC谱
musicSpectrum = musicSpectrum / max(musicSpectrum);

% 获取幅度
absMusicSpectrum = abs(musicSpectrum);

% 在图上标记信号的角度峰值
[pks, locs] = findpeaks(absMusicSpectrum);
figure;
plot(theta, 10*log10(absMusicSpectrum));
xlabel('角度（度）');
ylabel('归一化MUSIC谱（dB）');
title('PCA-MUSIC 结合使用');
hold on;
plot(theta(locs), 10*log10(pks), 'Marker', 'p', 'MarkerSize', 6, 'LineStyle', 'none', 'MarkerEdgeColor', [72/255, 192/255, 170/255]);
hold off;
