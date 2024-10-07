function ft = fun(pp, hd_num, net, pm_train, tm_train)

% 节点数
it_num = size(pm_train, 1);  % 输入层
ot_num = size(tm_train, 1);  % 输出层

w1 = pp(1 : it_num*hd_num);
b1 = pp(it_num*hd_num + 1 : it_num*hd_num + hd_num);
w2 = pp(it_num*hd_num + hd_num + 1 : it_num*hd_num + hd_num + hd_num*ot_num);
b2 = pp(it_num*hd_num + hd_num + hd_num*ot_num + 1 : it_num*hd_num + hd_num + hd_num*ot_num + ot_num);

net.Iw{1, 1} = reshape(w1, hd_num, it_num );
net.Lw{2, 1} = reshape(w2, ot_num, hd_num);
net.b{1}     = reshape(b1, hd_num, 1);
net.b{2}     = b2';
net = train(net, pm_train, tm_train);
t_sim1 = sim(net,pm_train );
tsim1   = vec2ind(t_sim1  );
t_train = vec2ind(tm_train);
% 适应度值
ft = 1 - sum(tsim1 == t_train)/length(tsim1);