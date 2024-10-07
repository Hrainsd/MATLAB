function x = parse(instr)
% 切割字符串
str_length = size(instr, 2);
x = blanks(str_length);
count = 1;
last = 0;
for i = 1 : str_length
  if instr(i) == ' '
    count = count + 1;
    x(count, :) = blanks(str_length);
    last = i;
  else
    x(count, i - last) = instr(i);
  end
end