% [P,d] = shortestpath(G,s,t,'Method',algorithm)
% 在图G中找到s节点到t节点的最短路径P和最短距离d
% dt = distances(G[,'Method',algorithm])返回任意两点的距离矩阵
% G = digraph([],[],weights[])有向图
% G = graph([],[],weights[])无向图
% highlight(p,P,'Edgecolor','b')表示高亮蓝色显示图p中路径P
A = [1 1 1 2 3 3 4 5 5 5 5 6 6 7 9 9];
B = [2 3 4 5 2 4 6 4 6 7 8 5 7 8 5 8];
W = [6 3 1 1 2 2 10 6 4 3 6 10 2 4 2 3];
G = digraph(A,B,W);
x = [-1 -0.5 -0.5 -0.5 0 0 0.25 0.5 0.5];
y = [0 0.5 0 -0.5 0.5 -0.5 -0.25 0 0.5];
dt = distances(G);
subplot(1,2,1);
plot(G);title('图论图');grid on;
subplot(1,2,2);
p = plot(G,'XData',x,'YData',y,'EdgeLabel',G.Edges.Weight,'EdgeColor','k');title('图论图');
[P,d] = shortestpath(G,1,8)
highlight(p,P,'EdgeColor','b');
