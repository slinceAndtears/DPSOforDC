# 1.每个节点随机初始化聚类中心点、拉格朗日乘数（初始化有什么规律吗）、u>0
# 2.在每个节点上，数据点分配给他最近的中心点。
# 3.给每个节点根据广播自己的中心点，以及给下一个邻居（节点编号加一的邻居）发送拉格朗日乘数。
# 4.每个节点更新中心点。
# 5.每个节点更新拉格朗日乘数。
# 重复2-5 直到达到收敛（最大迭代次数）
# next peer 意为下一个邻居,不是很懂什么意思（一跳的邻居）
# 对所有本地数据求出Dunn指标，然后求他们的平均值,不公平(对所有站点的数据进行汇总，然后用kmeans聚合得出结果)
# 问题：聚类之后，不同的中心点变成一样的了（原论文更新中心点的地方，除数是站点数量，现论文是处理数据总量）

from mpi4py import MPI
import numpy as np
from KMeans import initCentroid, DBIndex, Assign, DunnIndex, Kmeans
from DKmeans import Add
maxIte1 = 15  # 欧式距离的最大迭代次数
maxTie2 = 15  # 点对称距离的最大迭代次数
k = 8  # 聚类个数


def UpdateCentroidsAndLag(data, label, allCentroids, allLag, oldCentroids, oldLag, u, rank, size):
    length = len(data)
    dimension = len(oldCentroids[0])
    centroid = np.zeros([k, dimension], dtype=float)
    L = np.zeros([k, dimension], dtype=float)
    nextPeer = (rank + 1 + size) % size
    for i in range(k):
        # 更新中心点
        sumPoint = np.zeros(dimension, dtype=float)
        n = 0
        for j in range(length):
            if label[j] == i:
                sumPoint += data[j]
                n += 1
        sumCentroids = np.zeros(dimension, dtype=float)
        for j in range(size):
            sumCentroids += 2 * u * \
                oldCentroids[i] - oldLag[i] + allLag[(j - 1 + size) % size][i]
        centroid[i] = (sumPoint + sumCentroids) / (2 * u * size + n)
        # 更新拉格朗日乘数
        L = L + u * (centroid[i] - allCentroids[nextPeer][i])
    return centroid, L


def EucDistance(d1, d2):
    distance = 0.
    for i in range(len(d1)):
        distance += (d1[i] - d2[i]) * (d1[i] - d2[i])
    return np.sqrt(distance)


def SymDistance(d1, d2):
    distance = 0.
    return distance


def PointSymDistance(data, centroid):
    de = EucDistance(data, centroid)
    distance = 0.
    return distance


def GetFinalCentroid(allCentroids, size):
    finalcentroid = Add(allCentroids[0], allCentroids[1])
    for i in range(2, size):
        finalcentroid = Add(finalcentroid, allCentroids[i])
    centroids = Kmeans(k, finalcentroid)
    return centroids


def PSDKM():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.loadtxt('dataset/2d8c/2d8c%d.txt' % rank)
    allData = np.loadtxt('dataset/2d8c/2d8c.txt')
    dimension = len(data[0])
    length = len(data)
    centroids = initCentroid(k, data)
    #u = np.random.random_sample()
    u = 0.6
    print('this is process%d 初始化的L为' % rank)
    # 初始化拉格朗日乘数
    L = np.zeros([k, dimension], dtype=float)
    for i in range(k):
        for j in range(dimension):
            L[i][j] = centroids[i][j]
    allCentroids = {}
    allLag = {}
    # 欧式距离迭代
    for i in range(maxIte1+maxTie2):
        if i < maxIte1:
            label = Assign(centroids, data)
        else:
            label = np.zeros([1])
        allCentroids[rank] = centroids
        allLag[rank] = L
        # 向所有邻居发送中心点
        for j in range(size):
            if j != rank:
                comm.send(centroids, dest=j, tag=1)
                comm.send(L, dest=j, tag=2)
        # 接收所有邻居的中心点
        for j in range(size):
            if j != rank:
                allCentroids[j] = comm.recv(source=j, tag=1)
                allLag[j] = comm.recv(source=j, tag=2)
        centroids, L = UpdateCentroidsAndLag(
            data, label, allCentroids, allLag, centroids, L, u, rank, size)
    # 交给rank为0的进程对所有中心点进行汇总
    if rank == 0:
        finalCentroids = GetFinalCentroid(allCentroids, size)


if __name__ == "__main__":
    PSDKM()