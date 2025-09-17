import numpy as np
from scipy.optimize import linprog
from scipy.integrate import quad

# 飞机座位数 CASM RASM 再捕获率
aircraft_seats = [162, 200]
casms = [0.042, 0.044]
rasms = [0.15, 0.15]
recaps = [0.15, 0.15]

# 航线距离 平均值 标准差
infm = np.array([
    [2475, 175, 35], [2475, 182, 36], [2475, 145, 29], [2586, 178, 35],
    [2586, 195, 39], [2586, 162, 32], [740, 165, 33], [740, 182, 36],
    [740, 170, 34], [760, 191, 38], [760, 171, 34], [760, 165, 33],
    [1090, 198, 39], [1090, 182, 36], [1090, 168, 33], [187, 115, 23],
    [187, 146, 29], [187, 120, 24], [228, 135, 27], [228, 109, 21],
    [228, 98, 19], [2475, 150, 30], [2475, 145, 29], [2475, 125, 25],
    [2586, 148, 29], [2586, 138, 27], [2586, 121, 24], [740, 132, 26],
    [740, 129, 25], [740, 117, 23], [760, 168, 33], [760, 160, 32],
    [760, 191, 38], [1090, 165, 33], [1090, 184, 36], [1090, 192, 38],
    [187, 147, 29], [187, 135, 27], [187, 146, 29], [228, 105, 21],
    [228, 115, 23], [228, 118, 23]
])

# 计算成本
cost1 = np.zeros((42, 2))
cost2 = np.zeros((42, 2))
cost = np.zeros((42, 2))

for i, seat in enumerate(aircraft_seats):
    casm = casms[i]
    rasm = rasms[i]
    recap = recaps[i]
    for j in range(len(infm)):
        cost1[j, i] = round(casm * infm[j, 0] * seat, 2)

        # 定义积分函数
        integral_func = lambda x: (x - seat) * (1 / (np.sqrt(2 * np.pi) * infm[j, 2])) * np.exp(
            -(x - infm[j, 1]) ** 2 / (2 * infm[j, 2] ** 2))

        # 计算cost2的积分值
        integral_result = quad(integral_func, seat, np.inf)[0]
        cost2[j, i] = round(rasm * infm[j, 0] * round(integral_result, 2), 2)
        cost[j, i] = round(cost1[j, i] + round((1 - recap) * cost2[j, i], 2), 2)

cost = cost.flatten()

# 变量定义
f_values = cost
f_ind = []

for i in range(1, 43):
    for j in range(1, 3):
        f_ind.append(i * 10 + j)

f_ind = np.array(f_ind)
f = np.zeros(1262)
f[f_ind - 1] = f_values

intcon = list(range(1, 1263))

# 机队规模约束
A = np.zeros((2, 1262))
A[0, [480, 540, 600, 660, 720, 780, 840, 1260]] = 1
A[1, [481, 541, 601, 661, 720, 780, 840, 1261]] = 1
B = [9, 6]

# 航班覆盖约束
Aeq = np.zeros((210, 1262))
f_ind1 = np.reshape(f_ind, (-1, 2)).T

for i in range(42):
    Aeq[i, f_ind1[:, i] - 1] = 1

# 飞机平衡约束
# 制作第一列变量
ind_1 = []

for i in range(43, 127):
    for j in range(1, 3):
        ind_1.append(i * 10 + j)

ind_1 = np.reshape(ind_1, (-1, 2))

# 制作第二列变量
ind_2_1 = ind_1[:42, :].copy()
ind_2_2 = ind_1[42:, :].copy()

numGroups = ind_2_1.shape[0] // 6

for i in range(numGroups):
    groupIdx = slice(i * 6, (i + 1) * 6)
    ind_2_1[groupIdx] = np.vstack([ind_2_1[groupIdx][-1, :], ind_2_1[groupIdx][:-1, :]])

ind_2_2 = np.vstack([ind_2_2[-1, :], ind_2_2[:-1, :]])
ind_2 = np.vstack([ind_2_1, ind_2_2])

# 制作第三列变量
ind_3 = np.array([
    11, 21, 221, 31, 231, 241, 41, 251, 51, 61, 261, 271,
    161, 371, 171, 381, 181, 391, 71, 281, 81, 291, 91, 301,
    101, 311, 111, 321, 121, 331, 401, 191, 411, 201, 421, 211,
    131, 341, 141, 351, 361, 151, 11, 41, 161, 401, 251, 71,
    221, 371, 101, 191, 131, 311, 21, 51, 171, 281, 341, 411,
    81, 381, 111, 201, 141, 321, 181, 291, 351, 421, 31, 61,
    261, 231, 91, 121, 331, 361, 151, 211, 241, 271, 301, 391
])

# 合并三列变量
ind = np.column_stack((ind_1, ind_2, ind_3))

# 制作变量系数
coef = np.ones((84, 3))
coef[:, 1] = -1
coef[:, 2] = [
    1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1,
    1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1,
    -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1,
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1
]

for i in range(2):
    for j in range(84):
        Aeq[j + i * 84 + 42, [ind[j, i] - 1, ind[j, i + 2] - 1, ind[j, -1] - 1]] = coef[j, :]

Beq = np.zeros(210)
Beq[:42] = 1

lb = np.zeros(1262)
ub = np.ones(1262)
ub[ind_1[:, 0] - 1] = 9
ub[ind_1[:, 1] - 1] = 6

# 优化
result = linprog(f, A_ub=A, b_ub=B, A_eq=Aeq, b_eq=Beq, bounds=list(zip(lb, ub)), method='highs')

# 输出
ResultIndexes = np.where(result.x[:422] != 0)[0]
ResultValues = result.x[ResultIndexes]
Result = np.vstack((ResultIndexes + 1, ResultValues)).T
z = np.dot(f, result.x)
print(Result)

for i in range(len(ResultIndexes) // 2):
    print("变量：{}，取值：{} 变量：{}，取值：{}".format(ResultIndexes[2 * i] + 1, ResultValues[2 * i], ResultIndexes[2 * i + 1] + 1, ResultValues[2 * i + 1]))
print("目标函数值：{}".format(result.fun))
