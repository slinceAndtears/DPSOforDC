from mpi4py import MPI
import numpy as np
from KMeans import Kmeans
from KMeans import DBIndex
from KMeans import Assign
from KMeans import FitCH, Iindex
from Dunn import dunn_fast


def Add(d1, d2):
    l1 = len(d1)
    l2 = len(d2)
    d = len(d1[0])
    centroid = np.zeros([l1+l2, d])
    for i in range(l1):
        centroid[i] = d1[i]
    for i in range(l1, l1+l2):
        centroid[i] = d2[i-l1]
    return centroid


def DKmeans(k, data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == size-1:
        d = len(data[0])
        finalCentroid = np.zeros([k, d])
        for i in range(size-1):
            recvCentroid = comm.recv(source=i)
            if i == 0:
                centroid = recvCentroid
            else:
                centroid = Add(centroid, recvCentroid)
        finalCentroid = Kmeans(k, centroid)
        print('finish')
        filename = '10d10c_DKmeans.txt'
        storeResult(data, finalCentroid, filename)
    else:
        d = np.loadtxt('dataset/10d10c/10d10c%d.txt' % rank)
        # print(len(d))
        centroid = Kmeans(k, d)
        comm.send(centroid, dest=size - 1)


def storeResult(data, centroid, filename):  # 用于保存最终的结果
    label = Assign(centroid, data)
    f = open(filename, 'a')
    f.writelines('This is result\n')
    f.writelines(str(centroid)+'\n')

    f.writelines('This is Ch index\n')
    CHValue = FitCH(data, label, centroid)
    f.writelines(str(CHValue)+'\n')

    f.writelines('This is DB index\n')
    DBValue = DBIndex(data, label, centroid)
    f.writelines(str(DBValue)+'\n')

    f.writelines('This is I index\n')
    IValue = Iindex(data, label, centroid)
    f.writelines(str(IValue)+'\n')

    f.writelines('This is Dunn index\n')
    DunnValue = dunn_fast(data, label, centroid)
    f.writelines(str(DunnValue)+'\n')

    f.close()


if __name__ == "__main__":
    data = np.loadtxt('dataset/10d10c/10d10c.txt')
    DKmeans(10, data)
