from operator import le
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import random
from sklearn.datasets import make_blobs, make_circles
from sklearn import datasets
from scipy import stats
import pyradbas as pyb
import sklearn.neural_network as nn
#from FCM import fcm


def generate_dataset():  # 随机生成数据集
    data, label = make_blobs(n_samples=8000, n_features=10, centers=8, cluster_std=[
                             1.5, 1.6, 1.7, 1.7, 1.6, 1.5, 1.6, 1.6])
    np.savetxt('10d8c.txt', data)
    np.savetxt('label.txt', label, fmt='%d')
    pca = PCA(n_components=2)
    pca.fit(data)
    data = pca.fit_transform(data)
    plt.scatter(data[:, 0], data[:, 1], c=label, marker='.')
    plt.show()


def SendData(sum, l):  # s个数据，随机均分成l份
    d = np.zeros(sum, dtype=int)  # 每个数据的标签 属于哪个站点
    s = sum//l
    t = np.zeros(l, dtype=int)
    i = 0
    while i < s*l:
        r = random.randint(0, l-1)
        if t[r] < s:
            d[i] = r
            t[r] += 1
            i += 1
    if i <= sum:
        for j in range(i, sum):
            d[j] = random.randint(0, l-1)
            t[d[j]] += 1
    return d, t


def divideData1(sum, l):  # 按顺序分数据
    d = np.zeros(sum, dtype=int)
    s = np.zeros(l, dtype=int)
    tag = 0
    for i in range(sum):
        d[i] = tag
        s[tag] += 1
        tag += 1
        if tag == l:
            tag = 0
    return d, s


def divideData2(sum, l):
    d = np.zeros(sum, dtype=int)
    s = np.zeros(l, dtype=int)
    a = sum//l
    for i in range(l):
        for j in range(a):
            d[a*i+j] = i
            s[i] += 1
    if a*l < sum:
        for i in range(a*l, sum):
            d[i] = l-1
            s[l-1] += 1
    return d, s


def divide_data():
    N = 10  # 分成几份
    data = np.loadtxt('dataset/HTRU2/HTRU2.txt')
    l = len(data)
    d = len(data[0])
    [dis, sum] = SendData(l, N)
    for i in range(N):
        x = 0
        t = np.zeros((sum[i], d))
        for j in range(l):
            if dis[j] == i:
                t[x] = data[j]
                x += 1
        np.savetxt('dataset/HTRU2/HTRU2%d.txt' % i, t)


def show_data():
    data = np.loadtxt('dataset/ring-column/ring-column0.txt')
    #label=np.loadtxt('label.txt',dtype=int)
    label= dcpso(3,data)
    #data,label=datasets.make_moons(n_samples=1200,noise=0.08)
    #np.savetxt('half-ring-1200.txt',data)
    #np.savetxt('label.txt',label,fmt='%d')
    #pca = PCA(n_components=2)
    #pca.fit(data)
    #data = pca.fit_transform(data)
    plt.scatter(data[:, 0], data[:, 1],c=label, marker='.')
    plt.show()


def calculate_var():
    data0 = np.loadtxt('r1.txt')
    data1 = np.loadtxt('r2.txt')
    data2 = np.loadtxt('r3.txt')
    data3 = np.loadtxt('r4.txt')
    mt_levene=stats.levene(data0,data1,data2)
    #mt_friedman = stats.friedmanchisquare(data0,data1,data2,data3)
    mt_dpso_to_DK=stats.wilcoxon(data0,data1)
    mt_dpso_to_DSCA=stats.wilcoxon(data0,data2)
    mt_dpso_to_PSDKM=stats.wilcoxon(data0,data3)
    #nt_bartlett=stats.bartlett(data0,data1,data2)

    print('DPSO avg is:')
    print(np.mean(data0))
    print('DKM avg is:')
    print(np.mean(data1))
    print('DSCA avg is:')
    print(np.mean(data2))
    print("PSDKM avg is")
    print(np.mean(data3))

    print('DPSO std is:')
    print(np.std(data0))
    print('DKM std is:')
    print(np.std(data1))
    print('DSCA std is:')
    print(np.std(data2))
    print("PSDKM std is")
    print(np.std(data3))


    print('------std test-------')
    print(stats.f_oneway(data0,data1))
    print(stats.f_oneway(data0,data2))
    print(stats.f_oneway(data0,data3))
    #print(stats.bartlett(data0,data1,data2))
    print('-----avg test-----')
    #print(mt_levene)
    print('dpso to dk is')
    print(mt_dpso_to_DK)
    print('dpso to DSCA is')
    print(mt_dpso_to_DSCA)
    print('DPSO to PSDKM is')
    print(mt_dpso_to_PSDKM)


def Test5():
    # 计算方差
    # result=np.loadtxt('severCode/r3.txt',dtype=float)
    # result=np.array([1,2,3])
    print('xxxxx')
    # data,label=datasets.make_circles(n_samples=1200,noise=0.05,factor=0.2)
    bar1 = np.zeros([300, 2], dtype=float)
    f = open('circle.txt', 'a')
    for i in range(len(bar1)):
        bar1[i][0] = np.random.uniform(-1, 2)
        bar1[i][1] = np.random.uniform(1.2, 2)
        line = str(bar1[i])
        line = line.strip('[')
        line = line.strip(']')
        f.writelines(line)
        f.writelines('\n')
    #np.set_printoptions(suppress = True)
    # np.savetxt('circle.txt',data,fmt='%f')
    # np.savetxt('label.txt',label,fmt='%d')
    plt.scatter(bar1[:, 0], bar1[:, 1], marker='.')
    plt.show()


def Test6():
    l = 100
    I = np.zeros([l, 2])
    O = np.zeros(l)
    for i in range(l):
        I[i][0] = i
        I[i][1] = i+10
        O[i] = i+i+10
    model = nn.MLPRegressor(hidden_layer_sizes=(
        3,), activation='relu', solver='adam', max_iter=1000)
    model.fit(I, O)
    pre = np.array([2, 2], dtype=np.float)
    pre[0][0] = 10
    pre[0][1] = 20
    pre[1][0] = 20
    pre[1][1] = 40
    print(model.predict(pre))


if __name__ == "__main__":
    #Test5()
    #show_data()
    calculate_var()
