function pp = initializega(num, bounds, evalfn, evalops, opts)
%% 初始化参数
if nargin < 5
  opts = [1e-6, 1];
end
if nargin < 4
  evalops = [];
end

%% 编码
if any(evalfn < 48)    % M文件
  if opts(2) == 1      % 浮点数编码
    e_str = ['x=pp(i,1); pp(i,str_length)=', evalfn ';'];  
  else                 % 二进制编码
    e_str = ['x=b2f(pp(i,:),bounds,bits); pp(i,str_length)=', evalfn ';']; 
  end
else                   % not M文件
  if opts(2) == 1      % 浮点数编码
    e_str = ['[ pp(i,:) pp(i,str_length)]=' evalfn '(pp(i,:),[0 evalops]);']; 
  else                 % 二进制编码
    e_str = ['x=b2f(pp(i,:),bounds,bits);[x v]=' evalfn '(x,[0 evalops]); pp(i,:)=[f2b(x,bounds,bits) v];'];  
  end
end

%%  设置参数 
var_num = size(bounds, 1); 		             % 变量数
range     = (bounds(:, 2) - bounds(:, 1))';  % 可变范围

%%  编码
if opts(2) == 1                  % 二进制编码
  str_length = var_num + 1; 	 % 字符串长度
  pp = zeros(num, str_length); % 分配新种群
  pp(:, 1 : var_num) = (ones(num, 1) * range) .* (rand(num, var_num)) + (ones(num, 1) * bounds(:, 1)');
else                             % 浮点数编码
  bits = calcbits(bounds, opts(1));
  pp = round(rand(num, sum(bits) + 1));
end

%%  结果
for i = 1 : num
  eval(e_str);
end
