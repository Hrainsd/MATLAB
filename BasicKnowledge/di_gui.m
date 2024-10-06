function [jc] = di_gui(n)
%递归
%计算n的阶乘
if n == 0 || n == 1
    result = 1;
else
    result = n * di_gui(n-1);
end
jc = result;
end


