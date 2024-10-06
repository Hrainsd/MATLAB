% 鲸鱼优化算法
function [bestSolution, bestFitness] = Whale_Optimization_Algorithm_max_fun(objFunction, numVariables, numWhales, maxIterations, minValue, maxValue)

    % 参数设置
    a = 2; % 收缩系数
    maxA = 2; % 最大收缩系数
    minA = 0; % 最小收缩系数
    b = 1; % 振幅减小系数
    maxB = 1; % 最大振幅减小系数
    minB = 0; % 最小振幅减小系数
    numDimensions = numVariables; % 变量维度
    
    % 初始化鲸鱼群体
    whales = rand(numWhales, numDimensions) * (maxValue - minValue) + minValue; % 随机初始化鲸鱼位置
    fitnessValues = zeros(numWhales, 1);
    
    % 计算初始适应度
    for i = 1:numWhales
        fitnessValues(i) = objFunction(whales(i, :)); 
    end
    
    % 初始化最佳解和适应度
    [bestFitness, bestIndex] = max(fitnessValues); % 使用max来寻找最大值
    bestSolution = whales(bestIndex, :);
    
    % 开始优化
    for iteration = 1:maxIterations
        for i = 1:numWhales
            % 随机选择一个鲸鱼
            randWhaleIndex = randi(numWhales);
            
            % 更新鲸鱼位置
            A = (maxA - minA) * rand + minA;
            C = 2 * rand;
            l = (2 * rand) - 1;
            p = rand;
            
            if p < 0.5
                if abs(A) >= 1
                    randWhale = whales(randWhaleIndex, :);
                    d = abs(C * randWhale - whales(i, :));
                    newWhale = randWhale - A * d;
                else
                    randWhale = whales(randWhaleIndex, :);
                    newWhale = randWhale - A * (randWhale - whales(i, :));
                end
            else
                newWhale = bestSolution - A * l;
            end
            
            % 边界处理
            newWhale(newWhale < minValue) = minValue; % 将小于最小值的变量设置为最小值
            newWhale(newWhale > maxValue) = maxValue; % 将大于最大值的变量设置为最大值
            
            % 计算新适应度
            newFitness = objFunction(newWhale); 
            
            % 更新最佳解
            if newFitness > bestFitness % 使用大于号来判断是否更好
                bestFitness = newFitness;
                bestSolution = newWhale;
            end
            
            % 更新鲸鱼位置和适应度
            whales(i, :) = newWhale;
            fitnessValues(i) = newFitness;
        end
        
        % 更新振幅减小系数
        a = maxA - (iteration / maxIterations) * (maxA - minA);
        b = maxB - (iteration / maxIterations) * (maxB - minB);
    end
end