from cProfile import label
from operator import le
import numpy as np
import sys
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from FCM import fcm
from Dunn import dunn_fast
from queue import PriorityQueue
import matplotlib.pyplot as plt
# 欧几里得距离
knear = 2
maxIte = 20


def distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return np.sqrt(sum)


def Manhattan_distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += np.abs(d1[i]-d2[i])
    return sum


def Chebyshev_distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        if sum < np.abs(d1[i]-d2[i]):
            sum = np.abs(d1[i]-d2[i])
    return sum


def PointSymDistance(allData, point, centroid):
    eucDis = distance(point, centroid)
    psd = 0.
    SymData = centroid*2-point
    q = PriorityQueue()
    for i in range(len(allData)):
        dis = distance(allData[i], SymData)
        q.put(dis, dis)
    sum = 0.
    for i in range(knear):
        sum += q.get()
    return eucDis*(sum/knear)


def Distance1(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return sum
# 初始化中心点


def Assign_base_PSDistance(centroid, data):
    size = len(data)
    label = np.zeros(size, dtype=int)
    for i in range(size):
        min = sys.maxsize
        for j in range(len(centroid)):
            if(min > PointSymDistance(data, data[i], centroid[j])):
                min = PointSymDistance(data, data[i], centroid[j])
                label[i] = j
    return label


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


def Kmeans_basePSDistance(k, data):
    centroid = initCentroid(k, data)
    error = 0.
    label = Assign_base_PSDistance(centroid, data)
    for i in range(maxIte):
        print('ite: '+str(i))
        error = SequareError(centroid, data, label)
        centroid = getNewCentroid(label, data, k, centroid)
        label = Assign_base_PSDistance(centroid, data)
    return centroid


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
    label = Assign_base_PSDistance(centroid, data)
    while error != SequareError(centroid, data, label):
        error = SequareError(centroid, data, label)
        centroid = getNewCentroid(label, data, k, centroid)
        label = Assign_base_PSDistance(centroid, data)
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

    #f.writelines('This is Dunn index\n')
    #DunnValue = dunn_fast(data, label)
    #f.writelines(str(DunnValue)+'\n')

    f.close()


def getValidIndexResult():
    d = 2
    c = 2
    centroids = np.zeros([c, d], dtype=float)
    dataset = np.loadtxt('dataset/ring-round/ring-round.txt')
    x = 0
    y = 0
    k = 1
    f = open('centroid.txt', 'r')
    for line in f.readlines():
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.strip('\n')
        line = line.split()
        # print(line)
        line = [float(x) for x in line]
        a = np.array(line)
        for i in range(len(a)):
            centroids[x][y] = a[i]
            y += 1
            if y == d:
                y = 0
                x += 1
        if x == c and y == 0:
            print('第%d次的结果为:' % k)
            k += 1
            label = Assign_base_PSDistance(centroids, dataset)
            print('SH index is :')
            print(metrics.silhouette_score(dataset, label))
            print('Dunn index is')
            print(dunn_fast(dataset, label))
            label_test = Assign(centroids, dataset)
            print('test CH index is')
            print(FitCH(dataset, label_test, centroids))
            centroids = np.zeros([c, d], dtype=float)
            x = 0
            y = 0
    #print('CH index is :')
    #print(FitCH(dataset, label, centroids))
    # print(metrics.calinski_harabasz_score(dataset,label))
    #print('DB index is :')
    #print(DBIndex(dataset, label, centroids))
    # print(metrics.davies_bouldin_score(dataset,label))
    #print('I index is :')
    #print(Iindex(dataset, label, centroids))
    #print("Dunn Index is :")
    #print(dunn_fast(dataset, label))


def Test3():
    data = np.loadtxt('dataset/half-ring-1000/half-ring-1000.txt')
    k = 2
    print('finish')
    centroid, label = Kmeans_basePSDistance(k, data)
    #label = Assign_base_PSDistance(centroid, data)
    # print(DBIndex(data,label,centroid))
    #myindex = DunnIndex(data, label, k)
    # print('myindex')
    # print(myindex)
    centroid1 = Kmeans(k, data)
    label1 = Assign(centroid1, data)
    print('PS distance')
    print(metrics.silhouette_score(data, label))
    print(dunn_fast(data, label))
    print('eu distance')
    print(metrics.silhouette_score(data, label1))
    print(dunn_fast(data, label1))
    print(centroid)
    print(centroid1)
    plt.scatter(data[:, 0], data[:, 1], c=label, marker='.')
    plt.show()
    # print(dunn_fast(data,label))
    # print(dunn_fast(data,label1))


def test_DBSCAN():
    data = np.loadtxt('dataset/half-ring-1000/half-ring-1000.txt')
    label = DBSCAN(eps=0.1).fit_predict(data)
    print(label)
    print('DBSCAN')
    print(dunn_fast(data, label))
    print(metrics.silhouette_score(data, label))
    plt.scatter(data[:, 0], data[:, 1], c=label, marker='.')
    plt.show()


def Test1():
    data=np.loadtxt('dataset/10d20c/10d20c.txt')
    fileName='10d20cDSCA.txt'
    for i in range(20):
        centroid=fcm(data,20)
        label=Assign(centroid,data)
        storeResult(data,centroid,fileName)


def cluster_map():
    data = np.loadtxt('centroid.txt')
    #label = np.loadtxt('r1.txt', dtype=int)
    '''
    t=np.zeros([len(data),2],dtype=int)
    for i in range(len(data)):
        t[i][0]=data[i][0]
        t[i][1]=data[i][1]
    np.savetxt('r1.txt',t,fmt='%d')
    '''
    k=75
    model=KMeans(n_clusters=k).fit(data)
    label=model.labels_
    sum=np.zeros(k,dtype=int)
    for i in range(len(data)):
        sum[label[i]]+=1
    print(sum)
    print(np.max(sum))
    print(np.min(sum))
    
    plt.scatter(data[:, 0], data[:, 1], c=label, marker='.')
    plt.show()
    np.set_printoptions(suppress=True)
    np.savetxt('r1.txt', label, fmt='%d')
    

if __name__ == "__main__":
    Test1()
    # test_DBSCAN()
    #getValidIndexResult()
    #cluster_map()
