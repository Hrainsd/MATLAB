function [] = Floyd_path(W)
% 求各个节点之间的最短路径
[stdt,path] = Floyd(W);
[n,m] = size(W);
% 默认n>2
for i = 1:n
    for j = 1:n
        if i ~= j  
            Floyd_onepath(path,stdt,i,j);
            disp(' ')
        end
    end
end
end