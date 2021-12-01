import numpy as np
import sys
from sklearn import metrics
from sklearn.cluster import KMeans
from FCM import fcm
def distance(d1,d2):
    sum=0.
    for i in range(len(d1)):
        sum+=(d1[i]-d2[i])*(d1[i]-d2[i])
    return np.sqrt(sum)
def Distance1(d1,d2):
    sum=0.
    for i in range(len(d1)):
        sum+=(d1[i]-d2[i])*(d1[i]-d2[i])
    return sum
def initCentroid(k,data):
    size=len(data)
    d=len(data[0])
    Index=np.zeros(size,dtype=int)
    for i in range(size):
        Index[i]=i
    for i in range(size):
        r=np.random.randint(i,size)
        t=Index[i]
        Index[i]=Index[r]
        Index[r]=t
    result=np.zeros([k,d])
    for i in range(k):
        result[i]=data[Index[i]]
    return result
def Assign(centroid,data):
    size=len(data)
    label=np.zeros(size,dtype=int)
    for i in range(size):
        min=sys.maxsize
        for j in range(len(centroid)):
            if(min>distance(data[i],centroid[j])):
                min=distance(data[i],centroid[j])
                label[i]=j
    return label
def SequareError(centroid,data,label):
    error=0.
    for i in range(len(data)):
        error+=distance(data[i],centroid[label[i]])
    return error
def getNewCentroid(label,data,k,centroid):
    size=len(data)
    num=np.zeros(k,dtype=int)
    sum=np.zeros([k,len(data[0])])
    for i in range(size):
        sum[label[i]]+=data[i]
        num[label[i]]+=1
    for i in range(k):
        if num[i]!=0:
            sum[i]=sum[i]/num[i]
        else:
            sum[i]=centroid[i] 
    return sum
def Average(data,label,k,centroid):
    sum=0.
    s=0
    for i in range(len(data)):
        if label[i]==k:
            sum+=distance(data[i],centroid)
            s+=1
    if s!=0:
        return sum/s
    return 0
def DBIndex(data,label,centroid):
    k=len(centroid)
    sum=0.
    for i in range(k):
        max=0
        for j in range(k):
            if i!=j:
                ei=Average(data,label,i,centroid[i])
                ej=Average(data,label,j,centroid[j])
                Ri=(ei+ej)/distance(centroid[i],centroid[j])
                if max<Ri:
                    max=Ri
        sum+=max
    return sum/k
def Kmeans(k,data):
    centroid=initCentroid(k,data)
    error=0.
    label=Assign(centroid,data)
    while error!=SequareError(centroid,data,label):
        error=SequareError(centroid,data,label)
        centroid=getNewCentroid(label,data,k,centroid)
        label=Assign(centroid,data)
    return centroid
def kmeans(centroid,data):
    error=0.
    k=len(centroid)
    label=Assign(centroid,data)
    while error!=SequareError(centroid,data,label):
        error=SequareError(centroid,data,label)
        centroid=getNewCentroid(label,data,k,centroid)
        label=Assign(centroid,data)
    return centroid
def FitCH(Z,label,centroid):
    k=len(centroid)
    Nd=len(Z[0])
    Nc=len(Z)
    sum=np.zeros(k,dtype=int)
    Tsb=0.
    Tsw=0.
    m=np.zeros(Nd)
    for i in range(Nc):
        m+=Z[i]
        sum[label[i]]+=1
    for i in range(Nd):
        m[i]=m[i]/Nc
    for i in range(k):
        Tsb+=sum[i]*Distance1(centroid[i],m)
    for i in range(Nc):
        Tsw+=Distance1(Z[i],centroid[label[i]])
    return (Tsb/(k-1))/(Tsw/(Nc-k))
def Test3():
    data=np.loadtxt('n10d8c/10d8c.txt')
    centroid=Kmeans(8,data)
    label=Assign(centroid,data)
    print(DBIndex(data,label,centroid))
def Test1():
    for i in range(30):
        Test3()
if __name__ == "__main__":
    Test1()