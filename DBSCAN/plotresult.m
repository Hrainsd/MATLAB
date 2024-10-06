function plotresult(X, lbl)
    k = max(lbl);
    colors = hsv(k);
    figure;
    hold on;
    legends = {};
    for i = 0:k
        X_i = X(lbl == i, :);
        if i ~= 0
            style = 'o';
            markersize = 4;
            color = colors(i, :);
            legends{end + 1} = ['聚类' num2str(i)];
        else
            style = 'x';
            markersize = 4;
            color = 'k';
            if ~isempty(X_i)
                legends{end + 1} = '噪点';
            end
        end
        if ~isempty(X_i)
            plot(X_i(:, 1), X_i(:, 2), style, 'MarkerSize', markersize, 'Color', color, 'DisplayName', legends{end});
        end
    end
    hold off;
    
    axis equal;
    grid on;
    title('DBSCAN Clustering Result');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend(legends, 'Location', 'NorthEastOutside');
end