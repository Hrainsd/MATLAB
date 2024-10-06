function [x] = finding_roots(x1,x2,x3)
delta = x2^2-4*x1*x3;
x = [(-x2+sqrt(delta))/2*x1 , (-x2-sqrt(delta))/2*x1]
end

