from mpi4py import MPI
import numpy as np
def Initcenter(data,k):
    l=len(data)
    d=len(data[0])
    temp=np.zeros(l,dtype=int)
    centroid=np.zeros([k,d],dtype=np.float)
    for i in range(l):
        temp[i]=i
    for i in range(l-1):
        r=np.random.randint(i+1,l)
        temp[i],temp[r]=temp[r],temp[i]
    for i in range(k):
        centroid[i]=data[temp[i]]
    return centroid
def dkm():
    k=3
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()
    data=np.loadtxt('data/seeds/seeds%d.txt'%rank)
    centroid=Initcenter(data,k)#中心点
    u=np.random.random()
    y=np.zeros(k,dtype=np.float)#拉格朗日乘数
    recv={}#从其他节点接收的中心点
    recvY=np.zeros(k,dtype=np.float)#从相邻节点接收的u值
    for i in range(k):
        y=np.random.random()
    for i in range(size):
        if i!=rank:
            comm.send(centroid,dest=i)
        if i==(rank+1)%size:
            comm.send(y,dest=i)
    for i in range(size):
        if i!=rank:
            recv[i]=comm.recv(source=i)
def Test2():
    #data=np.loadtxt('data/seeds/seeds0.txt')
    #print(Initcenter(data,3))
    dic={}
    dic['A']='a'
    dic['B']='b'
    print(dic.get('B'))
if __name__ == "__main__":
    Test2()