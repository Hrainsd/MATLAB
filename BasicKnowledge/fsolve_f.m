function F = fsolve_f(x)
x0 = x(1);
y0 = x(2);
z0 = x(3);
F(1) = sin(x0)+y0+z0^2*exp(x0);
F(2) = x0+y0+z0;
F(3) = x0*y0*z0;
end
