function fin = maxgenterm(ops, ~, ~)
% 返回1时终止ga
currentge = ops(1);
maxge     = ops(2);
fin       = currentge >= maxge; 