import numpy as np
import sys
from sklearn import metrics
from sklearn.cluster import KMeans
from FCM import fcm
from Dunn import dunn_fast
# 欧几里得距离


def distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return np.sqrt(sum)


def Distance1(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return sum
# 初始化中心点


def initCentroid(k, data):
    size = len(data)
    d = len(data[0])
    Index = np.zeros(size, dtype=int)
    for i in range(size):
        Index[i] = i
    for i in range(size):
        r = np.random.randint(i, size)
        t = Index[i]
        Index[i] = Index[r]
        Index[r] = t
    result = np.zeros([k, d])
    for i in range(k):
        result[i] = data[Index[i]]
    return result
# 将每个数据点分配给对应的中心点-打标签


def Assign(centroid, data):
    size = len(data)
    label = np.zeros(size, dtype=int)
    for i in range(size):
        min = sys.maxsize
        for j in range(len(centroid)):
            if(min > distance(data[i], centroid[j])):
                min = distance(data[i], centroid[j])
                label[i] = j
    return label
# 目标函数


def SequareError(centroid, data, label):
    error = 0.
    for i in range(len(data)):
        error += distance(data[i], centroid[label[i]])
    return error
# 更新中心点


def getNewCentroid(label, data, k, centroid):
    size = len(data)
    num = np.zeros(k, dtype=int)
    sum = np.zeros([k, len(data[0])])
    for i in range(size):
        sum[label[i]] += data[i]
        num[label[i]] += 1
    for i in range(k):
        if num[i] != 0:
            sum[i] = sum[i]/num[i]
        else:
            sum[i] = centroid[i]
    return sum


def Average(data, label, k, centroid):
    sum = 0.
    s = 0
    for i in range(len(data)):
        if label[i] == k:
            sum += distance(data[i], centroid)
            s += 1
    if s != 0:
        return sum/s
    return 0
# DB指标


def DBIndex(data, label, centroid):
    k = len(centroid)
    sum = 0.
    for i in range(k):
        max = 0
        for j in range(k):
            if i != j:
                ei = Average(data, label, i, centroid[i])
                ej = Average(data, label, j, centroid[j])
                Ri = (ei+ej)/distance(centroid[i], centroid[j])
                if max < Ri:
                    max = Ri
        sum += max
    return sum/k
# 数据数据集和聚类个数K，输出K个中心点 用于常规聚类


def Kmeans(k, data):
    centroid = initCentroid(k, data)
    error = 0.
    label = Assign(centroid, data)
    while error != SequareError(centroid, data, label):
        error = SequareError(centroid, data, label)
        centroid = getNewCentroid(label, data, k, centroid)
        label = Assign(centroid, data)
    return centroid
# 输出K个中心点和数据集，输出K个中心点，用于DCPSO


def kmeans(data, centroid):
    error = 0.
    k = len(centroid)
    label = Assign(centroid, data)
    while error != SequareError(centroid, data, label):
        error = SequareError(centroid, data, label)
        centroid = getNewCentroid(label, data, k, centroid)
        label = Assign(centroid, data)
    return centroid
# CH指标值


def FitCH(Z, label, centroid):
    k = len(centroid)
    Nd = len(Z[0])
    Nc = len(Z)
    sum = np.zeros(k, dtype=int)
    Tsb = 0.
    Tsw = 0.
    m = np.zeros(Nd)
    for i in range(Nc):
        m += Z[i]
        sum[label[i]] += 1
    for i in range(Nd):
        m[i] = m[i]/Nc
    for i in range(k):
        Tsb += sum[i]*Distance1(centroid[i], m)
    for i in range(Nc):
        Tsw += Distance1(Z[i], centroid[label[i]])
    return (Tsb/(k-1))/(Tsw/(Nc-k))


def OneClusterDis(data):  # 计算一个簇内，两个点的最大距离
    maxDisdance = 0.
    length = len(data)
    dimension = len(data[0])
    for i in range(length-1):
        for j in range(i+1, length):
            maxDisdance = max(maxDisdance, distance(data[i], data[j]))
    return maxDisdance


def TwoClusterDis(data1, data2):  # 计算两个簇点的最小距离
    len1 = len(data1)
    len2 = len(data2)
    minDistance = sys.maxsize
    for i in range(len1):
        for j in range(len2):
            minDistance = min(minDistance, distance(data1[i], data2[j]))
    return minDistance


def DunnIndex(data, label, k):
    Di = 0.
    sum = np.zeros(k, dtype=int)
    length = len(data)
    dimension = len(data[0])
    d = {}
    result = 0.
    Dij = sys.maxsize
    for i in range(length):
        sum[label[i]] += 1
    for i in range(k):
        points = np.zeros([sum[i], dimension])
        index = 0
        for j in range(length):
            if label[j] == i:
                points[index] = data[j]
                index += 1
        d[i] = points
        Di = max(Di, OneClusterDis(points))
    for i in range(k-1):
        for j in range(i+1, k):
            Dij = min(Dij, TwoClusterDis(d[i], d[j]))
    result = Dij / Di
    return result


def Iindex(data, label, centroids):
    k = len(centroids)
    p = 2
    EK = SequareError(centroids, data, label)
    E1 = len(data)
    return ((1/k)*(E1/EK)*OneClusterDis(centroids))


def getValidIndexResult():
    data = np.loadtxt('centroid.txt')
    dataset=np.loadtxt('dataset/20d10c/20d10c.txt')
    d = 20
    c = 10
    centroids = np.zeros([10, 20], dtype=float)
    x = 0
    y = 0
    for i in range(len(data)):
        for j in range(len(data[0])):
            centroids[x][y] = data[i][j]
            y += 1
            if y == d:
                x += 1
                y = 0
    label=Assign(centroids,dataset)
    print('CH index is :')
    print(FitCH(dataset,label,centroids))
    print('DB index is :')
    print(DBIndex(dataset,label,centroids))
    print('I index is :')
    print(Iindex(dataset,label,centroids))
    print("Dunn Index is :")
    print(dunn_fast(dataset,label))

def Test3():
    data = np.loadtxt('dataset/compound/compound.txt')
    k = 8
    print('finish')
    centroid = Kmeans(k, data)
    label = Assign(centroid, data)
    # print(DBIndex(data,label,centroid))
    myindex = DunnIndex(data, label, k)
    print('myindex')
    print(myindex)


def Test1():
    for i in range(30):
        Test3()


if __name__ == "__main__":
    #Test3()
    getValidIndexResult()
