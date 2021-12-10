 #1.每个节点随机初始化聚类中心点、拉格朗日乘数（初始化有什么规律吗）、u>0
 #2.在每个节点上，数据点分配给他最近的中心点。
 #3.给每个节点根据广播自己的中心点，以及给下一个邻居（节点编号加一的邻居）发送拉格朗日乘数。
 #4.每个节点更新中心点。
 #5.每个节点更新拉格朗日乘数。
 # 重复2-5 直到达到收敛（最大迭代次数）

from mpi4py import MPI
import numpy as np
from KMeans import initCentroid
from KMeans import Assign
maxIte=10  #最大迭代次数
k=3#聚类个数

def UpdateCentroids(allCentroids,centroids,u):
    centroid = np.zeros([len(centroids),len(centroids[0])])
    return centroid

def UpdateLag():
    return 0

def PSDKM():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.loadtxt('dataset/seeds/seeds%d.txt'%rank)
    centroids = initCentroid(k,data)
    label = Assign(centroids,data)
    u = np.random.random_sample()
    L = np.random.random_sample()#拉格朗日乘数

    for i in range(maxIte):
        allCentroids = {}
        allCentroids[rank] = centroids
        allLag=np.zeros(size,dtype=float)
        allLag[rank]=L
        #向所有邻居发送中心点
        for i in range(size):
            if i!=rank:
                comm.send(centroids,dest=i, tag = 1)
                comm.send(L,dest=i, tag = 2)
        #接收所有邻居的中心点
        for i in range(size):
            if i!=rank:
                allCentroids[i] = comm.recv(source = i, tag = 1)
                allLag[i] = comm.recv(source = i,tag = 2)
        newCentroids=UpdateCentroids(allCentroids,centroids,u)
        newL=UpdateLag()

        
if __name__=="__main__":
    PSDKM()