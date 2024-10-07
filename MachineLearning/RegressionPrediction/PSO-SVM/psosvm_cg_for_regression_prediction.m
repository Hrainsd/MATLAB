function [best_CVmse, best_c, best_g, p_o] = psosvm_cg_for_regression_prediction(train_label, train, p_o)

% ������ʼ��
if nargin == 2
    p_o = struct('c1', 1.5, 'c2', 1.7, 'maxgen', 20, 'sizepop', 20, ...
                    'k', 0.6, 'wV', 1, 'wP', 1, 'v',5, ...
                     'popcmax', 10^2, 'popcmin', 10^(-1), 'popgmax', 10^3, 'popgmin', 10^(-2));
end

vcmax = p_o.k * p_o.popcmax; % �����ٶ����ֵ
vcmin = -vcmax ;

vgmax = p_o.k * p_o.popgmax;
vgmin = -vgmax ;

eps = 10^(-10); % ���������ֵ

% ������ʼ���Ӻ��ٶ�
for i = 1 : p_o.sizepop
    
    % ���������Ⱥ���ٶ�
    pp(i, 1) = (p_o.popcmax - p_o.popcmin) * rand + p_o.popcmin;  
    pp(i, 2) = (p_o.popgmax - p_o.popgmin) * rand + p_o.popgmin;
    v(i, 1)   = vcmax * rands(1, 1);  
    v(i, 2)   = vgmax * rands(1, 1);
    
    % �����ʼ��Ӧ��
    mm = [' -v ', num2str(p_o.v), ' -c ', num2str(pp(i, 1)), ' -g ', ...
        num2str(pp(i, 2)), ' -s 3 -p 0.1'];

    fitness(i) = svmtrain(train_label, train, mm);

end


[gb_fitness, bestindex] = min(fitness); % ��ʼ����ֵ�ͼ�ֵ��   
lc_fitness = fitness;                      
gb_x = pp(bestindex, :);                 
lc_x = pp;                                

% ÿһ����Ⱥ��ƽ����Ӧ��
af_gen = zeros(1, p_o.maxgen); 

%  ����Ѱ��
for i = 1 : p_o.maxgen
    for j = 1 : p_o.sizepop
        
        % �ٶȸ���        
        v(j, :) = p_o.wV * v(j, :) + p_o.c1 * rand * (lc_x(j, :) - pp(j, :)) + ...
            p_o.c2 * rand * (gb_x - pp(j, :));

        if v(j, 1) > vcmax
           v(j, 1) = vcmax;
        end

        if v(j, 1) < vcmin
           v(j, 1) = vcmin;
        end

        if v(j, 2) > vgmax
           v(j, 2) = vgmax;
        end

        if v(j, 2) < vgmin
           v(j, 2) = vgmin;
        end
        
        % ��Ⱥ����
        pp(j, :) = pp(j, :) + p_o.wP * v(j, :);
        
        if pp(j, 1) > p_o.popcmax
           pp(j, 1) = p_o.popcmax;
        end
        
        if pp(j, 1) < p_o.popcmin
           pp(j, 1) = p_o.popcmin;
        end
        
        if pp(j, 2) > p_o.popgmax
           pp(j, 2) = p_o.popgmax;
        end
        
        if pp(j, 2) < p_o.popgmin
           pp(j, 2) = p_o.popgmin;
        end
        
        % ����Ӧ���ӱ���
        if rand > 0.5
            k = ceil(2 * rand);

            if k == 1
               pp(j, k) = (20 - 1) * rand + 1;
            end

            if k == 2
               pp(j, k) = (p_o.popgmax - p_o.popgmin) * rand + p_o.popgmin;
            end     

        end
        
        % ��Ӧ��ֵ
        mm = [' -v ', num2str(p_o.v), ' -c ', num2str(pp(j, 1)), ' -g ', ...
            num2str(pp(j, 2)), ' -s 3 -p 0.1'];

        fitness(j) = svmtrain(train_label, train, mm);
        
        % �������Ÿ���
        if fitness(j) < lc_fitness(j)
           lc_x(j, :) = pp(j, :);
           lc_fitness(j) = fitness(j);
        end

        if fitness(j) == lc_fitness(j) && pp(j, 1) < lc_x(j, 1)
           lc_x(j, :) = pp(j, :);
           lc_fitness(j) = fitness(j);
        end        
        
        % Ⱥ�����Ÿ���
        if fitness(j) < gb_fitness
           gb_x = pp(j, :);
           gb_fitness = fitness(j);
        end

        if abs(fitness(j) - gb_fitness) <= eps && pp(j,1) < gb_x(1)
           gb_x = pp(j, :);
           gb_fitness = fitness(j);
        end
        
    end
    
    fit_gen(i) = gb_fitness;    
    af_gen(i) = sum(fitness) / p_o.sizepop;
end

%  ��Ӧ������
figure
plot(1 : length(fit_gen), fit_gen, 'b-', 'LineWidth', 1.5);
title('�����Ӧ������', 'FontSize', 13);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��', 'FontSize', 10);
xlim([1, length(fit_gen)])
grid on

%  ��ֵ
best_c = gb_x(1);
best_g = gb_x(2);
best_CVmse = fit_gen(p_o.maxgen);
