function pat_ge = nonunimutation(pat_ge, bounds, ops)
% 非均匀突变
cg = ops(1); 				      % 当代
mg = ops(3);                      % 最大代
bm = ops(4);                      % 形状参数
numvar = size(pat_ge, 2) - 1; 	  
mpoint = round(rand*(numvar - 1)) + 1;  
d = round(rand); 			      % 突变方向
if d 					          % 向上限突变
  new_value = pat_ge(mpoint) + dlt(cg, mg, bounds(mpoint, 2) - pat_ge(mpoint), bm);
else 					          % 向下限突变
  new_value = pat_ge(mpoint) - dlt(cg, mg, pat_ge(mpoint) - bounds(mpoint, 1), bm);
end
pat_ge(mpoint) = new_value; 	  % 得到子代