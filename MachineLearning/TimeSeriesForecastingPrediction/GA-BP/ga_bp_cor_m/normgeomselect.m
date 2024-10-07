function newpp = normgeomselect(oldpp, options)
% 归一化几何分布排序选择函数。
% 交叉 选择 排序
q = options(2); 				    
e = size(oldpp, 2); 			    
n = size(oldpp, 1);  		        
newpp = zeros(n, e); 		        
fit = zeros(n, 1); 		            
x = zeros(n,2); 			        
x(:, 1) = (n : -1 : 1)'; 	        
[~, x(:, 2)] = sort(oldpp(:, e));  
% 参数
r = q/(1 - (1 - q) ^ n); 			            % 归一化
fit(x(:, 2)) = r*(1 - q).^(x(:, 1) - 1); 	% 选择概率
fit = cumsum(fit); 			                    % 累积概率

% 初始化
rnum = sort(rand(n, 1)); 			            
fitinm = 1;                                      % 初始化
newinm = 1; 			                            
while newinm <= n 				                % 获得n个新个体
  if(rnum(newinm) < fit(fitinm)) 		
    newpp(newinm, :) = oldpp(fitinm, :); 	    % 选择
    newinm = newinm + 1; 			                % 寻找下一个个体
  else
    fitinm = fitinm + 1; 			                % 潜在个体
  end
end