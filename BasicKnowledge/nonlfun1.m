function [c,ceq] = nonlfun(x)
x1 = x(1);
x2 = x(2);
x3 = x(3);
c = [-x1-x2^2+2;x2+2*x3^2-3];
ceq = [-x1^2+x2-x3^2;x1+x2^2+x3^2-20];
end
