import numpy as np
from scipy.optimize import linprog

# 超参数设置
f = np.zeros(472)
values = [5, 5, 8, 8, 7, 7, 4, 4, 1, 1, 5, 5, 8, 8, 3, 3, 4, 4, 4, 4]
indices = [130, 131, 140, 141, 230, 231, 240, 241, 350, 351, 360, 361, 370, 371, 450, 451, 460, 461, 470, 471]
f[indices] = values

intcon = list(range(472))

A = np.zeros((5, 472));
A[0, [140, 141]] = 1;
A[1, [240, 241]] = 1;
A[2, [450, 451]] = 1;
A[3, [460, 461]] = 1;
A[4, [470, 471]] = 1;
B = 50 * np.ones(5);
Aeq = np.zeros((14, 472))
Aeq[0, [130, 140]] = 1
Aeq[1, [230, 240]] = 1
Aeq[2, [130, 230, 350, 360, 370]] = [1, 1, -1, -1, -1]
Aeq[3, [140, 240, 450, 460, 470]] = [1, 1, -1, -1, -1]
Aeq[4, [350, 450]] = 1
Aeq[5, [360, 460]] = 1
Aeq[6, [370, 470]] = 1
Aeq[7, [131, 141]] = 1
Aeq[8, [231, 241]] = 1
Aeq[9, [131, 231, 351, 361, 371]] = [1, 1, -1, -1, -1]
Aeq[10, [141, 241, 451, 461, 471]] = [1, 1, -1, -1, -1]
Aeq[11, [351, 451]] = 1
Aeq[12, [361, 461]] = 1
Aeq[13, [371, 471]] = 1
Beq = [40, 50, 0, 0, 30, 30, 30, 35, 25, 0, 0, 20, 30, 10]

bounds = [(0, 0) for _ in range(472)]
ub_ind = [130, 140, 230, 240, 350, 360, 370, 450, 460, 470, 131, 141, 231, 241, 351, 361, 371, 451, 461, 471]
ub = [40, 40, 50, 50, 90, 90, 90, 90, 90, 90, 35, 35, 25, 25, 60, 60, 60, 60, 60, 60]

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
