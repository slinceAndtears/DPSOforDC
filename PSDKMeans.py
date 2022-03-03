# 1.每个节点随机初始化聚类中心点、拉格朗日乘数（初始化有什么规律吗）、u>0
# 2.在每个节点上，数据点分配给他最近的中心点。
# 3.给每个节点根据广播自己的中心点，以及给下一个邻居（节点编号加一的邻居）发送拉格朗日乘数。
# 4.每个节点更新中心点。
# 5.每个节点更新拉格朗日乘数。
# 重复2-5 直到达到收敛（最大迭代次数）
# next peer 意为下一个邻居,不是很懂什么意思（一跳的邻居）
# 对所有本地数据求出Dunn指标，然后求他们的平均值,不公平(对所有站点的数据进行汇总，然后用kmeans聚合得出结果)
# 问题：聚类之后，不同的中心点变成一样的了（原论文更新中心点的地方，除数是站点数量，现论文是处理数据总量）

import sys
from mpi4py import MPI
import numpy as np
from KMeans import Assign_base_PSDistance, initCentroid, DBIndex, Assign, DunnIndex, Kmeans, storeResult, Kmeans_basePSDistance
from queue import PriorityQueue
from DKmeans import Add
maxIte = 40
maxIte1 = int(maxIte * 0.8)  # 欧式距离的最大迭代次数
maxTie2 = int(maxIte * 0.2)  # 点对称距离的最大迭代次数
k = 3  # 聚类个数
knear = 4  # 点对称距离中的参数 此处和原始论文中一直
data_name = 'ring-column'


def PartitionBaseSymDis(centroids, data):
    nn_dis = 0.
    for i in range(len(data)):
        min_dis = sys.maxsize
        for j in range(len(data)):
            if i != j and min_dis > EucDistance(data[i], data[j]):
                min_dis = EucDistance(data[i], data[j])
        if nn_dis < min_dis:
            nn_dis = min_dis
    label = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        min = sys.maxsize
        min_index = 0
        for j in range(len(centroids)):
            symdis = PointSymDistance(data, data[i], centroids[j])
            if min > symdis:
                min = symdis
                min_index = j
        if min/EucDistance(data[i], centroids[min_index]) <= nn_dis:
            label[i] = min_index
        else:
            min = sys.maxsize
            min_ind = 0
            for j in range(len(centroids)):
                eu_dis = EucDistance(data[i], centroids[j])
                if min > eu_dis:
                    min = eu_dis
                    min_ind = j
            label[i] = min_ind
    return label


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
        L[i] = L[i] + u * (centroid[i] - allCentroids[nextPeer][i])
    return centroid, L


def EucDistance(d1, d2):
    distance = 0.
    for i in range(len(d1)):
        distance += (d1[i] - d2[i]) * (d1[i] - d2[i])
    return np.sqrt(distance)


def PointSymDistance(allData, point, centroid):
    eucDis = EucDistance(point, centroid)
    psd = 0.
    SymData = centroid*2-point
    q = PriorityQueue()
    for i in range(len(allData)):
        dis = EucDistance(allData[i], SymData)
        q.put(dis, dis)
    sum = 0.
    for i in range(knear):
        sum += q.get()
    return eucDis*(sum/knear)


def GetFinalCentroid(allCentroids, size):
    finalcentroid = Add(allCentroids[0], allCentroids[1])
    for i in range(2, size):
        finalcentroid = Add(finalcentroid, allCentroids[i])
    centroids = Kmeans_basePSDistance(k, finalcentroid)
    return centroids


def PSDKM():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.loadtxt('dataset/'+data_name+'/'+data_name+'%d.txt' % rank)
    allData = np.loadtxt('dataset/'+data_name+'/'+data_name+'.txt')
    dimension = len(data[0])
    length = len(data)
    centroids = initCentroid(k, data)
    #u = np.random.random_sample()
    u = 6
    print('this is process%d 初始化的L为' % rank)
    # 初始化拉格朗日乘数
    L = np.zeros([k, dimension], dtype=float)
    for i in range(k):
        for j in range(dimension):
            L[i][j] = centroids[i][j]
    allCentroids = {}
    allLag = {}
    for i in range(maxIte1+maxTie2):
        if i < maxIte1:  # 先使用欧式距离迭代
            label = Assign(centroids, data)
        else:  # 后续再使用点对称距离
            label = PartitionBaseSymDis(centroids, data)
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
        filename = 'PSDKM'+data_name+'.txt'
        storeResult(allData, finalCentroids, filename)


if __name__ == "__main__":
    PSDKM()
