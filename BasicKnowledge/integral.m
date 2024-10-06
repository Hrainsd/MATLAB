function [integral] = integral(a)
n = 1;
sum = 1;
while n <= a
    sum = sum * n;
    n = n + 1;
end
integral = sum
