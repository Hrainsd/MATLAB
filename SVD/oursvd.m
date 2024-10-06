function [real_ratio,compressed_A] = oursvd(A,ratio)
[n,m] = size(A);% (n = m)
[U,S,V] = svd(A);
s = diag(S);
for i =1:n
    if sum(s(1:i))/sum(s) > ratio
        break
    end
end
real_ratio = sum(s(1:i))/sum(s);
compressed_A = U(:,1:i)*S(1:i,1:i)*V(:,1:i)'; % 压缩后的矩阵
end