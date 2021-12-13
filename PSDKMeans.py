 #1.每个节点随机初始化聚类中心点、拉格朗日乘数（初始化有什么规律吗）、u>0
 #2.在每个节点上，数据点分配给他最近的中心点。
 #3.给每个节点根据广播自己的中心点，以及给下一个邻居（节点编号加一的邻居）发送拉格朗日乘数。
 #4.每个节点更新中心点。
 #5.每个节点更新拉格朗日乘数。
 # 重复2-5 直到达到收敛（最大迭代次数）
 # next peer 意为下一个邻居

from mpi4py import MPI
import numpy as np
from KMeans import initCentroid, DBIndex, Assign, DunnIndex
maxIte=10  #最大迭代次数
k=3#聚类个数

def UpdateCentroidsAndLag(data, label, allCentroids, allLag, oldCentroids, oldLag, u, rank, size):
    length = len(data)
    dimension = len(oldCentroids[0])
    centroid = np.zeros([k, dimension], dtype = float)
    L = np.zeros([k, dimension], dtype = float)
    nextPeer = (rank - 1 + size) % size
    for i in range(k):
        #更新中心点
        sumPoint = np.zeros(dimension, dtype = float)
        n = 0
        for j in range(length):
            if label[j] == i:
                sumPoint += data[j]
                n+=1
        sumCentroids = np.zeros(dimension, dtype = float)
        for j in range(size):
            sumCentroids += 2 * u * oldCentroids[i] - oldLag[i] + allLag[j][i]
        centroid[i] = (sumPoint + sumCentroids) / (2 * u * len(data) + n)
        #更新拉格朗日乘数
        L = L + u * (centroid[i] - allCentroids[nextPeer][i])
    return centroid, L

def PSDKM():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.loadtxt('dataset/seeds/seeds%d.txt'%rank)
    allData = np.loadtxt('dataset/seeds/seeds.txt')
    dimension = len(data[0])
    length = len(data)
    centroids = initCentroid(k,data)
    label = Assign(centroids,data)
    u = np.random.random_sample()
    #u = 0.6
    print('this is process%d 初始化的L为'%rank)
    #初始化拉格朗日乘数
    L = np.zeros([k,dimension], dtype = float)
    for i in range(k):
        for j in range(dimension):
            L[i][j]=centroids[i][j]
    for i in range(maxIte):
        label = Assign(centroids, data)
        allCentroids = {}
        allCentroids[rank] = centroids
        allLag = {}
        allLag[rank] = L
        #向所有邻居发送中心点
        for j in range(size):
            if j!= rank:
                comm.send(centroids, dest = j, tag = 1)
                comm.send(L, dest = j, tag = 2)
        #接收所有邻居的中心点
        for j in range(size):
            if j!= rank:
                allCentroids[j] = comm.recv(source = j, tag = 1)
                allLag[j] = comm.recv(source = j, tag = 2)
        
        centroids,L = UpdateCentroidsAndLag(data, label, allCentroids, allLag, centroids, L , u, rank, size)
    #问题：聚类之后，不同的中心点变成一样的了
        if rank==0:
            print('this is ite %d'%i)
            print(centroids)
        #print(DBIndex(allData, Assign(centroids, allData), centroids))
        #print(allLag)

def Test1():
    a=np.zeros([2,2])
    a[0][0]=1
    b=np.zeros(2)
    b=a[0]
    a[0][0]=2
    print(b)

if __name__=="__main__":
    PSDKM()
    #Test1()