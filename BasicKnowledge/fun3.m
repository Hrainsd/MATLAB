function f = fun3(x)
f = zeros(10,1);
f(1) = abs(x(1) - 1) + abs(x(2) - 2); 
f(2) = abs(x(1) - 4) + abs(x(2) - 10); 
f(3) = abs(x(1) - 3) + abs(x(2) - 8); 
f(4) = abs(x(1) - 5) + abs(x(2) - 18); 
f(5) = abs(x(1) - 9) + abs(x(2) - 1); 
f(6) = abs(x(1) - 12) + abs(x(2) - 4); 
f(7) = abs(x(1) - 6) + abs(x(2) - 5); 
f(8) = abs(x(1) - 20) + abs(x(2) - 10); 
f(9) = abs(x(1) - 17) + abs(x(2) - 8); 
f(10) = abs(x(1) - 8) + abs(x(2) - 9); 
end
