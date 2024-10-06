function [] = Floyd_onepath(path,stdt,i,j)
% 求节点i到节点j的最短路径
if path(i,j) == j   
    if dist(i,j) == Inf
        disp([num2str(i),'不能到达',num2str(j)])
    else
        disp([num2str(i),'到',num2str(j),'的最短路径为：'])
        disp([num2str(i),'→',num2str(j)])
        disp(['最短距离为:',num2str(stdt(i,j))])
    end
else
    t = path(i,j);
    op = [num2str(i),'→'];
    while t ~= j  
        op = [op , num2str(t) , '→' ]; 
        t = path(t,j);
    end
    op = [op , num2str(j)];
    disp([num2str(i),'到',num2str(j),'的最短路径为：'])
    disp(op)
    disp(['最短距离为：',num2str(stdt(i,j))])
end
end