function [best_CVacy, best_c, best_g, p_opt] = psosvm_cg_for_classification_prediction(t_train, p_train, p_opt)

%%  ������ʼ��
if nargin == 2
    p_opt = struct('c1', 1.5, 'c2', 1.7, 'maxgen', 10, 'sizepop', 10, ...
        'k', 0.6, 'wV', 1, 'wP', 1, 'v', 5, ...
        'popcmax', 10, 'popcmin', 10^(-1), 'popgmax', 10, 'popgmin', 10^(-1));
end

%%  ��������ٶ�
vcmax = p_opt.k * p_opt.popcmax;
vcmin = -vcmax;
vgmax = p_opt.k * p_opt.popgmax;
vgmin = -vgmax;

%%  �����ֵ
eps = 10^(-10);

%%  ��Ⱥ��ʼ��
for i = 1 : p_opt.sizepop
    
    % ���������Ⱥ���ٶ�
    pp(i, 1) = (p_opt.popcmax - p_opt.popcmin) * rand + p_opt.popcmin;
    pp(i, 2) = (p_opt.popgmax - p_opt.popgmin) * rand + p_opt.popgmin;
    v(i, 1) = vcmax * rands(1, 1);
    v(i, 2) = vgmax * rands(1, 1);
    
    % �����ʼ��Ӧ��
    mm = [' -v ', num2str(p_opt.v), ' -c ',num2str(pp(i, 1)), ' -g ', num2str(pp(i, 2))];
    fts(i) = (100 - svmtrain(t_train, p_train, mm)) / 100;
end

%%  ��ʼ����ֵ�ͼ�ֵ��
[gb_fts, best_ind] = min(fts);   % ȫ�ּ�ֵ
lc_fts = fts;                      % ���弫ֵ��ʼ��
gb_x = pp(best_ind, :);                 % ȫ�ּ�ֵ��
lc_x = pp;                                % ���弫ֵ���ʼ��

%%  ƽ����Ӧ��
avgfts_g = zeros(1, p_opt.maxgen);

%%  ����Ѱ��
for i = 1 : p_opt.maxgen
    for j = 1 : p_opt.sizepop
        
       % �ٶȸ���
        v(j, :) = p_opt.wV * v(j, :) + p_opt.c1 * rand * (lc_x(j, :) ...
            - pp(j, :)) + p_opt.c2 * rand * (gb_x - pp(j, :));
        
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
        pp(j, :) = pp(j, :) + p_opt.wP * v(j, :);

        if pp(j, 1) > p_opt.popcmax
           pp(j, 1) = p_opt.popcmax;
        end

        if pp(j, 1) < p_opt.popcmin
           pp(j, 1) = p_opt.popcmin;
        end

        if pp(j, 2) > p_opt.popgmax
           pp(j, 2) = p_opt.popgmax;
        end

        if pp(j, 2) < p_opt.popgmin
           pp(j, 2) = p_opt.popgmin;
        end
        
       % ����Ӧ���ӱ���
        if rand > 0.5
            k = ceil(2 * rand);

            if k == 1
                pp(j, k) = (20 - 1) * rand + 1;
            end
            
            if k == 2
                pp(j, k) = (p_opt.popgmax - p_opt.popgmin) * rand + p_opt.popgmin;
            end

        end
        
       % ��Ӧ��ֵ
       mm = [' -v ', num2str(p_opt.v), ' -c ', num2str(pp(j, 1)), ' -g ', num2str(pp(j, 2))];
       fts(j) = (100 - svmtrain(t_train, p_train, mm)) / 100;
        
       % �������Ÿ���
        if fts(j) < lc_fts(j)
            lc_x(j, :) = pp(j, :);
            lc_fts(j) = fts(j);
        end
        
        if abs(fts(j)-lc_fts(j)) <= eps && pp(j, 1) < lc_x(j, 1)
            lc_x(j, :) = pp(j, :);
            lc_fts(j) = fts(j);
        end
        
       % Ⱥ�����Ÿ���
        if fts(j) < gb_fts
            gb_x = pp(j, :);
            gb_fts = fts(j);
        end
        
        if abs(fts(j) - gb_fts) <= eps && pp(j, 1) < gb_x(1)
            gb_x = pp(j, :);
            gb_fts = fts(j);
        end
        
    end
    
    % ƽ����Ӧ�Ⱥ������Ӧ��
    f_g(i) = gb_fts;
    avgfts_g(i) = sum(fts) / p_opt.sizepop;

end

%%  ��Ӧ������
figure
plot(1 : length(f_g), f_g, 'b-', 'LineWidth', 1.5);
title ('��Ӧ������', 'FontSize', 13);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��', 'FontSize', 10);
grid on;

%%  ����ֵ��ֵ
best_c = gb_x(1);
best_g = gb_x(2);
best_CVacy = (1 - f_g(p_opt.maxgen)) * 100;