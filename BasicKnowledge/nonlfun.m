function [c,ceq] = nonlfun(x)
x1 = x(1);
x2 = x(2);
c = (x1-1)^2-x2;
ceq = [];
end

