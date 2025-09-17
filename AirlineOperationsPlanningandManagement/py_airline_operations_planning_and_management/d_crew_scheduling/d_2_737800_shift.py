import numpy as np
from scipy.io import loadmat
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# 读取 .mat 文件
data = loadmat(r"D:\Matlab\Shu_Xue_Jian_Mo\AO\intlinprog参数.mat")

# 提取数据
f_LunBan = data['f_LunBan'].flatten()  # 目标函数系数
A_LunBan = data['A_LunBan']  # 不等式约束系数
B_LunBan = data['B_LunBan'].flatten()  # 不等式约束的右侧
Aeq_LunBan = data['Aeq_LunBan']  # 等式约束系数
Beq_LunBan = data['Beq_LunBan'].flatten()  # 等式约束的右侧
intcon_LunBan = list(range(len(f_LunBan)))  # 假设所有变量都是整数
lb_LunBan = [0] * len(f_LunBan)  # 下界
ub_LunBan = [1] * len(f_LunBan)  # 上界

# 创建整数线性规划问题
problem = LpProblem("Integer_Linear_Programming", LpMinimize)

# 创建变量，限制为整数
x = [LpVariable(f"x{i}", lowBound=lb_LunBan[i], upBound=ub_LunBan[i], cat='Integer') for i in range(len(f_LunBan))]

# 目标函数
problem += lpSum(f_LunBan[i] * x[i] for i in range(len(f_LunBan))), "Objective"

# 不等式约束
for i in range(len(B_LunBan)):
    problem += lpSum(A_LunBan[i][j] * x[j] for j in range(len(x))) <= B_LunBan[i], f"Inequality_{i}"

# 等式约束
for i in range(len(Beq_LunBan)):
    problem += lpSum(Aeq_LunBan[i][j] * x[j] for j in range(len(x))) == Beq_LunBan[i], f"Equality_{i}"

# 解决问题
problem.solve()

# 输出结果
print("状态:", LpStatus[problem.status])
print('最优解:', value(problem.objective))
print('最优变量值:', [value(var) for var in x])

# 找到非零变量的索引
non_zero_indices = [i for i in range(len(x)) if value(x[i]) != 0]
print('轮班的数目:', len(non_zero_indices))
print('轮班的索引:', non_zero_indices)

