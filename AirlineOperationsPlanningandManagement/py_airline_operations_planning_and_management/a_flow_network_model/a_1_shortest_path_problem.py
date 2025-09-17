import numpy as np
from scipy.optimize import linprog

# 超参数设置
f = np.zeros(910)
values = [70, 63, 56, 25, 19, 73, 50, 79, 25, 29, 69, 61, 19,
          29, 67, 45, 85, 18, 67, 69, 54, 87, 18, 72, 52,
          51, 97, 17, 31, 72, 17, 15, 31, 15, 69]
indices = [11, 12, 13, 22, 23, 24, 25, 26, 31, 33, 34, 35,
           41, 42, 44, 45, 48, 55, 56, 57, 58, 509,
           64, 66, 67, 68, 609, 77, 78, 709, 86, 88, 96,
           97, 909]
f[indices] = values

intcon = list(range(910))

A = None
B = None
Aeq = np.zeros((10, 910))
Aeq[0, [11, 12, 13]] = 1
Aeq[1, [11, 31, 41, 22, 23, 24, 25, 26]] = [1, 1, 1, -1, -1, -1, -1, -1]
Aeq[2, [12, 22, 42, 31, 33, 34, 35]] = [1, 1, 1, -1, -1, -1, -1]
Aeq[3, [13, 23, 33, 41, 42, 44, 45, 48]] = [1, 1, 1, -1, -1, -1, -1, -1]
Aeq[4, [24, 34, 44, 64, 55, 56, 57, 58, 509]] = [1, 1, 1, 1, -1, -1, -1, -1, -1]
Aeq[5, [25, 35, 45, 55, 64, 66, 67, 68, 609]] = [1, 1, 1, 1, -1, -1, -1, -1, -1]
Aeq[6, [26, 56, 66, 86, 96, 77, 78, 709]] = [1, 1, 1, 1, 1, -1, -1, -1]
Aeq[7, [57, 67, 77, 97, 86, 88]] = [1, 1, 1, 1, -1, -1]
Aeq[8, [48, 58, 68, 78, 88, 96, 97, 909]] = [1, 1, 1, 1, 1, -1, -1, -1]
Aeq[9, [509, 609, 709, 909]] = [1, 1, 1, 1]
Beq = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

bounds = [(0, 1) for _ in range(910)]

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
