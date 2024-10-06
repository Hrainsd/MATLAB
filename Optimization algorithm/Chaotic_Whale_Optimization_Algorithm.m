function [bestSolution, bestFitness] = Chaotic_Whale_Optimization_Algorithm(objFunction, numVariables, numWhales, maxIterations, minValue, maxValue, optimizationType)
    % 参数设置
    shrinkageFactor = 2; % 收缩系数
    maxShrinkage = 2; % 最大收缩系数
    minShrinkage = 0; % 最小收缩系数
    amplitudeFactor = 1; % 振幅减小系数
    maxAmplitude = 1; % 最大振幅减小系数
    minAmplitude = 0; % 最小振幅减小系数
    temperature = 1; % 初始温度
    coolingRate = 0.95; % 降温速率
    numDimensions = numVariables; % 变量维度
    
    % 初始化鲸鱼群体
    whales = rand(numWhales, numDimensions) * (maxValue - minValue) + minValue; % 随机初始化鲸鱼位置
    fitnessValues = zeros(numWhales, 1);
    
    % 计算初始适应度
    for i = 1:numWhales
        fitnessValues(i) = objFunction(whales(i, :));
    end
    
    % 初始化最佳解和适应度
    if strcmp(optimizationType, 'min')
        [bestFitness, bestIndex] = min(fitnessValues);
    elseif strcmp(optimizationType, 'max')
        [bestFitness, bestIndex] = max(fitnessValues);
    else
        error('Invalid optimization type. Use ''min'' or ''max''.');
    end
    bestSolution = whales(bestIndex, :);
    
    % 开始优化
    for iteration = 1:maxIterations
        for i = 1:numWhales
            % 随机选择一个鲸鱼
            randWhaleIndex = randi(numWhales);
            
            % 更新鲸鱼位置
            shrinkage = (maxShrinkage - minShrinkage) * rand + minShrinkage;
            C = 2 * rand;
            l = (2 * rand) - 1;
            p = rand;
            
            if p < 0.5
                if abs(shrinkage) >= 1
                    randWhale = whales(randWhaleIndex, :);
                    d = abs(C * randWhale - whales(i, :));
                    newWhale = randWhale - shrinkage * d;
                else
                    randWhale = whales(randWhaleIndex, :);
                    newWhale = randWhale - shrinkage * (randWhale - whales(i, :));
                end
            else
                newWhale = bestSolution - shrinkage * l;
            end
            
            % 边界处理
            newWhale(newWhale < minValue) = minValue;
            newWhale(newWhale > maxValue) = maxValue;
            
            % 计算新适应度
            newFitness = objFunction(newWhale);
            
            % 模拟退火：根据Metropolis准则接受或拒绝新解
            if (strcmp(optimizationType, 'min') && (newFitness < fitnessValues(i) || (rand() < exp((fitnessValues(i) - newFitness) / temperature)))) || ...
               (strcmp(optimizationType, 'max') && (newFitness > fitnessValues(i) || (rand() < exp((newFitness - fitnessValues(i)) / temperature))))
                whales(i, :) = newWhale;
                fitnessValues(i) = newFitness;
                
                % 更新最佳解
                if (strcmp(optimizationType, 'min') && newFitness < bestFitness) || ...
                   (strcmp(optimizationType, 'max') && newFitness > bestFitness)
                    bestFitness = newFitness;
                    bestSolution = newWhale;
                end
            end
        end
        
        % 更新温度
        temperature = temperature * coolingRate;
        
        % 更新振幅减小系数
        shrinkageFactor = maxShrinkage - (iteration / maxIterations) * (maxShrinkage - minShrinkage);
        amplitudeFactor = maxAmplitude - (iteration / maxIterations) * (maxAmplitude - minAmplitude);
    end
end