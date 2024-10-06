A = input('请输入需要进行奇异值分解的原矩阵：');
ratio = input('请输入需要保留的特征比例（例如：0.9）：');
[real_ratio,compressed_A] = oursvd(A,ratio)
