import numpy as np
from scipy.optimize import linprog

# 超参数设置
f = np.zeros(45)
values = [-1]
indices = [11]
f[indices] = values

intcon = list(range(45))

A = None
B = None
Aeq = np.zeros((3, 45))
Aeq[0, [11, 22, 23]] = [1, -1, -1]
Aeq[1, [22, 34]] = [1, -1]
Aeq[2, [23, 44]] = [1, -1]
Beq = [0, 0, 0]

bounds = [(0, 0) for _ in range(45)]
ub_ind = [11, 22, 23, 34, 44]
ub = [3, 2, 3, 1, 2]

for index, ub in zip(ub_ind, ub):
    bounds[index] = (bounds[index][0], ub)

# 优化
result = linprog(c=f, A_ub=A, b_ub=B, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='highs')

# 输出
x = result.x
ResultIndexes = np.where(x != 0)[0]
ResultValues = x[ResultIndexes]
Result = np.vstack((ResultIndexes + 1, ResultValues)).T

for i in range(len(ResultIndexes)):
    print("变量：{}，取值：{}".format(ResultIndexes[i] + 1, ResultValues[i]))
print("目标函数值：{}".format(-result.fun))
