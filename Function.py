import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import random
from sklearn.datasets import make_blobs
from scipy import stats
import pyradbas as pyb
import sklearn.neural_network as nn
#from FCM import fcm
def Test1():
    data,label=make_blobs(n_samples=8000,n_features=10,centers=8,cluster_std=[1.5,1.6,1.7,1.7,1.6,1.5,1.6,1.6])
    np.savetxt('10d8c.txt',data)
    np.savetxt('label.txt',label,fmt='%d')
    pca=PCA(n_components=2)
    pca.fit(data)
    data=pca.fit_transform(data)
    plt.scatter(data[:,0],data[:,1],c=label,marker='.')
    plt.show()
def SendData(sum,l):#s个数据，随机均分成l份
    d=np.zeros(sum,dtype=int)#每个数据的标签 属于哪个站点
    s=sum//l
    t=np.zeros(l,dtype=int)
    i=0
    while i<s*l:
        r=random.randint(0,l-1)
        if t[r]<s:
            d[i]=r
            t[r]+=1
            i+=1    
    if i<=sum:
        for j in range(i,sum):
            d[j]=random.randint(0,l-1)
            t[d[j]]+=1
    return d,t
def divideData1(sum,l):#按顺序分数据
    d=np.zeros(sum,dtype=int)
    s=np.zeros(l,dtype=int)
    tag=0
    for i in range(sum):
        d[i]=tag
        s[tag]+=1
        tag+=1
        if tag==l:
            tag=0
    return d,s
def divideData2(sum,l):
    d=np.zeros(sum,dtype=int)
    s=np.zeros(l,dtype=int)
    a=sum//l
    for i in range(l):
        for j in range(a):
            d[a*i+j]=i
            s[i]+=1
    if a*l<sum:
        for i in range(a*l,sum):
            d[i]=l-1
            s[l-1]+=1
    return d,s
def Test3():
    N=8#分成几份
    data=np.loadtxt('D:/PyWorkSpace/n10d10c/10d10c.txt')
    l=len(data)
    d=len(data[0])
    [dis,sum]=SendData(l,N)
    for i in range(N):
        x=0
        t=np.zeros((sum[i],d))
        for j in range(l):
            if dis[j]==i:
                t[x]=data[j]
                x+=1
        np.savetxt('D:/PyWorkSpace/n10d10c/10d10c%d.txt'%i,t)
def Test2():
    data=np.loadtxt('D:/PyWorkSpace/severCode/newdata/10d8c/10d8c.txt')
    #label=np.loadtxt('D:/PyWorkSpace/severCode/data/10d10c/label.txt',dtype=int)
    pca=PCA(n_components=2)
    pca.fit(data)
    data=pca.fit_transform(data)
    plt.scatter(data[:,0],data[:,1],c='black',marker='.')
    plt.show()
def Test4():
    data0=np.loadtxt('severCode/r1.txt')
    data1=np.loadtxt('severCode/r2.txt')
    data2=np.loadtxt('severCode/r3.txt')
    #mt=stats.kruskal(data0,data1,data2)
    mt=stats.median_test(data0,data1,data2)
    #mt=stats.f_oneway(data0,data1,data2)
    #mt=stats
    #ks=stats.kstest(data0,data1)
    #mt=stats.ks_2samp(data0,data1)
    #mt=stats.wilcoxon(data0,data2)
    #print(fm.pvalue)
    print(mt)
def Test5():
    #计算方差
    #result=np.loadtxt('severCode/r3.txt',dtype=float)
    #result=np.array([1,2,3])
    print('xxxxx')
def Test6():
    l=100
    I=np.zeros([l,2])
    O=np.zeros(l)
    for i in range(l):
        I[i][0]=i
        I[i][1]=i+10
        O[i]=i+i+10
    model=nn.MLPRegressor(hidden_layer_sizes=(3,),activation='relu',solver='adam',max_iter=1000)
    model.fit(I,O)
    pre=np.array([2,2],dtype=np.float)
    pre[0][0]=10
    pre[0][1]=20
    pre[1][0]=20
    pre[1][1]=40
    print(model.predict(pre))
if __name__ == "__main__":   
    Test5()