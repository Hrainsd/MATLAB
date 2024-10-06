function [lbl, inie] = dbscan(X, epsilon, min_pts)
    c = 0;
    n = size(X, 1);
    lbl = zeros(n, 1);
    d   = pdist2(X, X);
    visited = false(n, 1);
    inie    = false(n, 1);
function neighbors = RegionQuery(i)
    neighbors = find(d(i, :) <= epsilon);
end
function ExpandCluster(i, neighbors, C)
    stack = neighbors; 
    while ~isempty(stack)
        j = stack(1);
        stack(1) = [];
        if ~visited(j)
            visited(j) = true;
            neighbors2 = RegionQuery(j);
            if numel(neighbors2) >= min_pts
                stack = [stack neighbors2]; 
            end
        end
        if lbl(j) == 0
            lbl(j) = C;
        end
    end
end
    for i = 1:n
        if ~visited(i)
            visited(i) = true;
            neighbors = RegionQuery(i);
            if numel(neighbors) < min_pts
                inie(i) = true;
            else
                c = c + 1;
                ExpandCluster(i, neighbors, c);
            end
        end
    end
end