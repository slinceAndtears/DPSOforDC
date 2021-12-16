import numpy as np
import sys
import copy
from sklearn import metrics


def Distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return np.sqrt(sum)


def Initialize_U(l, k):  # 初始化矩阵U
    U = np.zeros([l, k])
    for i in range(l):
        sum = 0.
        for j in range(k):
            U[i][j] = np.random.randint(1, 100)
            sum += U[i][j]
        for j in range(k):
            U[i][j] = U[i][j]/sum
    return U


def normalize_U(U):  # 归一化U
    for i in range(len(U)):
        sum = 0.
        for j in range(len(U[0])):
            sum += U[i][j]
        for j in range(len(U[0])):
            U[i][j] = U[i][j]/sum
    return U


def Update_U(data, U, centroid):
    m = 2
    c = 2./(m-1)
    for i in range(len(U)):
        for j in range(len(U[0])):
            sum = 0.
            for k in range(len(centroid)):
                t = Distance(data[i], centroid[j]) / \
                    Distance(data[i], centroid[k])
                sum += t**c
            U[i][j] = 1./sum
    return U


def calculateCentroid(data, U, c):
    m = 2
    d = len(data[0])  # 数据维度
    l = len(data)  # 数据集长度
    centroid = np.zeros([c, d])  # 新中心点
    for j in range(c):
        sum = 0.
        for i in range(l):
            temp = U[i][j]**m
            centroid[j] += temp*data[i]
            sum += temp
        centroid[j] = centroid[j]/sum
    return centroid


def Assign(centroid, data):
    size = len(data)
    label = np.zeros(size, dtype=int)
    for i in range(size):
        min = sys.maxsize
        for j in range(len(centroid)):
            if(min > Distance(data[i], centroid[j])):
                min = Distance(data[i], centroid[j])
                label[i] = j
    return label


def Delete(centroid, x):
    cent = np.zeros([len(centroid)-1, len(centroid[0])])
    for i in range(x):
        cent[i] = centroid[i]
    for i in range(x, len(centroid)-1):
        cent[i] = centroid[i+1]
    return cent


def fcm(data, c):  # 改为使用最大迭代次数
    maxItera = 70
    l = len(data)
    U = Initialize_U(l, c)
    for i in range(maxItera):
        centroid = calculateCentroid(data, U, c)
        U = Update_U(data, U, centroid)
    # 获取每个数据点的标签
    U = normalize_U(U)
    label = np.zeros(l, dtype=int)
    for i in range(l):
        max = U[i][0]
        tag = 0
        for j in range(len(U[0])):
            if max < U[i][j]:
                max = U[i][j]
                tag = j
        label[i] = tag
    # 求对应的中心点
    sum = np.zeros(c, dtype=int)
    cent = np.zeros([c, len(data[0])], dtype=np.float)
    for i in range(l):
        cent[label[i]] += data[i]
        sum[label[i]] += 1
    temp = 0
    for i in range(c):
        if sum[i] == 0:
            temp += 1
        else:
            cent[i] = cent[i]/sum[i]
    centroid = np.zeros([c-temp, len(data[0])], dtype=np.float)
    temp = 0
    for i in range(c):
        if sum[i] != 0:
            centroid[temp] = cent[i]
            temp += 1
    return centroid


def Test1():
    data = np.loadtxt('dataset/2d10c/2d10c0.txt')
    centre, label = fcm(data, 10)
    print(len(centre))
    print(metrics.davies_bouldin_score(data, label))


def Test2():
    for i in range(5):
        Test1()


if __name__ == "__main__":
    Test2()

# 最大迭代次数