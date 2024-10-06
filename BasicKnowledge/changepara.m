function [result1,result2] = changepara(a,b)
%测试可变参数

result1 = 0;
result2 = 0;
if nargin == 1 && nargout == 1
    result1 = a
elseif nargin == 2 && nargout == 2
    result1 = a
    result2 = b;
elseif nargin == 1 && nargout == 2
    result1 = a
    result2 = a
else
    result1 = a*b
end
