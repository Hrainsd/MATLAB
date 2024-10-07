function [C1, C2] = artovr(p1, p2, ~, ~)
% Arith交叉
% 产生子代
a = rand;
C1 = p1*a + p2*(1 - a);
C2 = p1*(1 - a) + p2*a;