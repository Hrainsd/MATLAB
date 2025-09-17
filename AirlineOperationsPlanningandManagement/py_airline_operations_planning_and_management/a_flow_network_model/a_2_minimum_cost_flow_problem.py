import numpy as np
from scipy.optimize import linprog

# 超参数设置
f = np.zeros(47)
values = [5, 8, 7, 4, 1, 5, 8, 3, 4, 4]
indices = [12, 13, 22, 23, 34, 35, 36, 44, 45, 46]
f[indices] = values

intcon = list(range(47))

A = None
B = None
Aeq = np.zeros((7, 47))
Aeq[0, [12, 13]] = 1
Aeq[1, [22, 23]] = 1
Aeq[2, [12, 22, 34, 35, 36]] = [1, 1, -1, -1, -1]
Aeq[3, [13, 23, 44, 45, 46]] = [1, 1, -1, -1, -1]
Aeq[4, [34, 44]] = 1
Aeq[5, [35, 45]] = 1
Aeq[6, [36, 46]] = 1
Beq = [75, 75, 0, 0, 50, 60, 40]

bounds = [(0, 0) for _ in range(47)]
ub_ind = [12, 13, 22, 23, 34, 35, 36, 44, 45, 46]
ub = [75, 50, 75, 50, 150, 150, 150, 50, 50, 50]

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
print("目标函数值：{}".format(result.fun))
