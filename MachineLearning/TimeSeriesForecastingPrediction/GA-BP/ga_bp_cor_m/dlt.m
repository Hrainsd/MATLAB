function dif = dlt(ct, mt, c, h)
% 非均匀分布  
r = ct / mt;
if (r > 1)
    r = 0.99;
end
dif = c*(rand * (1 - r))^h;