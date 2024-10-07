function [x, endpp, bpp, treinf] = ga(bounds, evalfn, evalops, startpp, opts, ...
termfn, termops, selectfn, selectops, xoverfns, xoverops, mutfns, mutops)
%% 初始化参数
n = nargin;
if n < 2 || n == 6 || n == 10 || n == 12
  disp('Insufficient arguements') 
end
% 默认eva
if n < 3 
  evalops = [];
end
% 默认参数
if n < 5
  opts = [1e-6, 1, 0];
end
if isempty(opts)
  opts = [1e-6, 1, 0];
end

%% 判断m文件
if any(evalfn < 48)
  % 浮点数编码 
  if opts(2) == 1
    e1str = ['x=c1; c1(str_length)=', evalfn ';'];  
    e2str = ['x=c2; c2(str_length)=', evalfn ';']; 
  % 二进制编码
  else
    e1str = ['x=b2f(endpp(j,:),bounds,bits); endpp(j,str_length)=', evalfn ';'];
  end
else
  % 浮点数编码
  if opts(2) == 1
    e1str = ['[c1 c1(str_length)]=' evalfn '(c1,[ge evalops]);'];  
    e2str = ['[c2 c2(str_length)]=' evalfn '(c2,[ge evalops]);'];
  % 二进制编码
  else
    e1str=['x=b2f(endpp(j,:),bounds,bits);[x v]=' evalfn ...
	'(x,[ge evalops]); endpp(j,:)=[f2b(x,bounds,bits) v];'];  
  end
end

%% 默认终止inf
if n < 6
  termops = 100;
  termfn = 'maxGenTerm';
end

%% 默认变异inf
if n < 12
  % 浮点数编码
  if opts(2) == 1
  mutfns = 'boundaryMutation multiNonUnifMutation nonUnifMutation unifMutation';
    mutops = [4, 0, 0; 6, termops(1), 3; 4, termops(1), 3;4, 0, 0];
  % 二进制编码
  else
    mutfns = 'binaryMutation';
    mutops = 0.05;
  end
end

%% 默认交叉inf
if n < 10
  % 浮点数编码
  if opts(2) == 1
    xoverfns = 'arithXover heuristicXover simpleXover';
    xoverops = [2, 0; 2, 3; 2, 0];
  % 二进制编码
  else
    xoverfns = 'simpleXover';
    xoverops = 0.6;
  end
end

%% 轮盘赌
if n < 9
  selectops = [];
end

%% 默认选择inf
if n < 8
  selectfn = 'normGeomSelect';
  selectops = 0.08;
end

%% 默认终止inf
if n < 6
  termops = 100;
  termfn = 'maxGenTerm';
end

%% 初始种群
if n < 4
  startpp = [];
end

%% 随机种群
if isempty(startpp)
  startpp = initializega(80, bounds, evalfn, evalops, opts(1: 2));
end

%% 二进制编码
if opts(2) == 0
  bits = calcbits(bounds, opts(1));
end

%% 设置参数
xoverfns   = parse(xoverfns);
mutfns     = parse(mutfns);
str_length = size(startpp, 2); 	          
var_num    = str_length - 1; 	         % 变量数
ppsize     = size(startpp,1); 	         % 种群个体数
endpp      = zeros(ppsize, str_length);  % 第二种群矩阵
numxovers  = size(xoverfns, 1);             
nummuts    = size(mutfns, 1); 		      
epsilon    = opts(1);                       
oval       = max(startpp(:, str_length)); 
bfound     = 1; 			                 
fin        = 0;                            
ge         = 1; 			                 
coltre     = (nargout > 3); 		        
floatga    = opts(2) == 1;                
dpy        = opts(3);                      

%% 精英模型
while(~fin)
  [bval, bidx] = max(startpp(:, str_length));           
  best =  startpp(bidx, :);
  if coltre
    treinf(ge, 1) = ge; 		                     
    treinf(ge, 2) = startpp(bidx,  str_length);     
    treinf(ge, 3) = mean(startpp(:, str_length));   
    treinf(ge, 4) = std(startpp(:,  str_length)); 
  end
  
  %% 最优值
  if ( (abs(bval - oval) > epsilon) || (ge==1))
    
  % 更新显示
    if dpy
      fprintf(1, '\n%d %f\n', ge, bval);          
    end

  % 更新种群
    if floatga
      bpp(bfound, :) = [ge, startpp(bidx, :)]; 
    else
      bpp(bfound, :) = [ge, b2f(startpp(bidx, 1 : var_num), bounds, bits)...
	  startpp(bidx, str_length)];
    end

    bfound = bfound + 1;                      
    oval = bval;                              
  else
    if dpy
      fprintf(1,'%d ',ge);	                  
    end
  end
%% 选择种群
  endpp = feval(selectfn, startpp, [ge, selectops]);
  
  % 运行模型
  if floatga
    for i = 1 : numxovers
      for j = 1 : xoverops(i, 1)
          a = round(rand * (ppsize - 1) + 1); 	     
	      b = round(rand * (ppsize - 1) + 1); 	     
	      cfn = deblank(xoverfns(i, :)); 	         
	      [c1, c2] = feval(cfn, endpp(a, :), endpp(b, :), bounds, [ge, xoverops(i, :)]);
          
          if c1(1 : var_num) == endpp(a, (1 : var_num)) 
	         c1(str_length) = endpp(a, str_length);
	      elseif c1(1:var_num) == endpp(b, (1 : var_num))
	         c1(str_length) = endpp(b, str_length);
          else
             eval(e1str);
          end

          if c2(1 : var_num) == endpp(a, (1 : var_num))
	          c2(str_length) = endpp(a, str_length);
	      elseif c2(1 : var_num) == endpp(b, (1 : var_num))
	          c2(str_length) = endpp(b, str_length);
          else
	          eval(e2str);
          end

          endpp(a, :) = c1;
          endpp(b, :) = c2;
      end
    end

    for i = 1 : nummuts
      for j = 1 : mutops(i, 1)
          a = round(rand * (ppsize - 1) + 1);
          c1 = feval(deblank(mutfns(i, :)), endpp(a, :), bounds, [ge, mutops(i, :)]);
          if c1(1 : var_num) == endpp(a, (1 : var_num)) 
              c1(str_length) = endpp(a, str_length);
          else
              eval(e1str);
          end
          endpp(a, :) = c1;
      end
    end

%% 遗传算子概率模型
  else 
    for i = 1 : numxovers
        cfn = deblank(xoverfns(i, :));
        cp = find((rand(ppsize, 1) < xoverops(i, 1)) == 1);

        if rem(size(cp, 1), 2) 
            cp = cp(1 : (size(cp, 1) - 1)); 
        end
        cp = reshape(cp, size(cp, 1) / 2, 2);

        for j = 1 : size(cp, 1)
            a = cp(j, 1); 
            b = cp(j, 2); 
            [endpp(a, :), endpp(b, :)] = feval(cfn, endpp(a, :), endpp(b, :), ...
                bounds, [ge, xoverops(i, :)]);
        end
    end

    for i = 1 : nummuts
        m_n = deblank(mutfns(i, :));
        for j = 1 : ppsize
            endpp(j, :) = feval(m_n, endpp(j, :), bounds, [ge, mutops(i, :)]);
            eval(e1str);
        end
    end

  end
  
  % 更新
  ge = ge + 1;
  fin = feval(termfn, [ge, termops], bpp, endpp); 
  startpp = endpp; 			                      
  [~, bidx] = min(startpp(:, str_length));        
  startpp(bidx, :) = best; 		                  
  
end
[bval, bidx] = max(startpp(:, str_length));

%% 显示结果
if dpy 
  fprintf(1, '\n%d %f\n', ge, bval);	  
end

%% 二进制编码
x = startpp(bidx, :);
if opts(2) == 0
  x = b2f(x, bounds,bits);
  bpp(bfound, :) = [ge, b2f(startpp(bidx, 1 : var_num), bounds, bits)...
      startpp(bidx, str_length)];
else
  bpp(bfound, :) = [ge, startpp(bidx, :)];
end

%% 结果
if coltre
  treinf(ge, 1) = ge; 		                      % 当前迭代次数
  treinf(ge, 2) = startpp(bidx, str_length);      % 最佳适应度
  treinf(ge, 3) = mean(startpp(:, str_length));   % 平均适应度
end