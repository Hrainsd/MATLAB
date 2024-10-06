function F = fminsearch_(X)
x = X(1);
y = X(2);
z = X(3);
F = x+y^2/(4*x)+z^2/y+2/z;
end

