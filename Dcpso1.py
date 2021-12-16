import numpy as np
import random
import sys
from KMeans import kmeans
from KMeans import DBIndex
from KMeans import Assign
s = 30
c1 = 1.49
c2 = 1.49
w = 0.82
MaxV = 255


def distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return np.sqrt(sum)


def FitnessFunction(data, label, centroid, Np):
    intra = 0.
    inter = 0.
    sum = 0
    k = len(centroid)
    for i in range(Np):
        sum += distance(data[i], centroid[label[i]])
    intra = sum/Np
    for i in range(k-1):
        for j in range(i+1, k):
            if inter < distance(centroid[i], centroid[j]) and distance(centroid[i], centroid[j]) != 0:
                inter = distance(centroid[i], centroid[j])
    return ((9*np.random.normal(2, 1)+1))*(intra/inter)


def init(Nc):
    S = np.zeros([s, Nc])
    V = np.zeros([s, Nc])
    Pini = 0.8
    for i in range(s):
        for j in range(Nc):
            rk = np.random.randint(1, 100)/100.
            if rk < Pini:
                S[i][j] = 1
            else:
                S[i][j] = 0
            V[i][j] = (np.random.randint(0, 100)/10.)-5
    return S, V


def UpdateParticle(S, V, Pbest, Gbest, Nc):
    for i in range(s):
        for j in range(Nc):
            r1 = np.random.randint(1, 100)/100.
            r2 = np.random.randint(1, 100)/100.
            V[i][j] += w*V[i][j]+c1*r1 * \
                (Pbest[i][j]-S[i][j])+c2*r2*(Gbest[j]-S[i][j])
            if V[i][j] > MaxV:
                V[i][j] = MaxV
            if V[i][j] < -MaxV:
                V[i][j] = -MaxV
            sig = 1/(1+1/np.exp(V[i][j]))
            Rk = np.random.randint(1, 100)/100.
            if Rk >= sig:
                S[i][j] = 0
            else:
                S[i][j] = 1
    return S, V


def initCentroid(data, Nc):
    size = len(data)
    d = len(data[0])
    Index = np.zeros(size, dtype=int)
    M = np.zeros([Nc, d])
    for i in range(size):
        Index[i] = i
    for i in range(size-1):
        r = np.random.randint(i+1, size)
        t = Index[i]
        Index[i] = Index[r]
        Index[r] = t
    for i in range(Nc):
        M[i] = data[Index[i]]
    return M


def getMt(S, M):
    Nc = len(M)
    Nd = len(M[0])
    sum = 0
    for i in range(Nc):
        if S[i] == 1:
            sum += 1
    if sum < 2:
        sum = 2
    Mt = np.zeros([sum, Nd])
    t = 0
    for i in range(Nc):
        if S[i] == 1:
            Mt[t] = M[i]
            t += 1
    for i in range(t, sum):
        r = np.random.randint(0, Nc)
        while S[r] != 0:
            r = np.random.randint(0, Nc)
        Mt[t] = M[r]
        S[r] = 1
        t += 1
    return Mt, S


def UpdateM(data, Mt, Nc):
    if len(Mt) != Nc:
        Np = len(data)
        Nd = len(data[0])
        t = Nc-len(Mt)
        M = np.zeros([Nc, Nd])
        Index = np.zeros(Np, dtype=int)
        for i in range(Np):
            Index[i] = i
        for i in range(1, Np-1):
            r = np.random.randint(i+1, Np)
            Index[i], Index[r] = Index[r], Index[i]
        for i in range(len(Mt)):
            M[i] = Mt[i]
        for i in range(t):
            M[i+len(Mt)] = data[Index[i]]
        return M
    else:
        return Mt


def delete(M, j):
    l = len(M)
    d = len(M[0])
    Mt = np.zeros((l-1, d))
    a = 0
    for i in range(l):
        if i != j:
            Mt[a] = M[i]
            a += 1
    return Mt


def testRepeat(M):
    i = 0
    while i < len(M)-1:
        j = i+1
        while j < len(M):
            if (M[i] == M[j]).all():
                M = delete(M, j)
            j += 1
        i += 1
    return M


def calculate(data, S, centroid):
    Np = len(data)
    F = np.zeros(s)
    for i in range(s):
        [Mt, P] = getMt(S[i], centroid)
        if len(Mt) >= 2:  # 如果删除多了
            Mt = testRepeat(Mt)
        label = Assign(Mt, data)
        F[i] = FitnessFunction(data, label, Mt, Np)
        S[i] = P
    return F, S


def getBestParticle(S, Pbest, PbestF, Gbest, GbestF, F, Nc):
    min = sys.maxsize
    tag = 0
    for i in range(s):
        if F[i] < PbestF[i]:
            PbestF[i] = F[i]
            for j in range(Nc):
                Pbest[i][j] = S[i][j]
        if min > F[i]:
            tag = i
            min = F[i]
    if min < GbestF:
        GbestF = min
        for i in range(Nc):
            Gbest[i] = S[tag][i]
    return Pbest, PbestF, Gbest, GbestF


def Loop1(data, M, Nc):
    [S, V] = init(Nc)
    Pbest = np.zeros([s, Nc])
    PbestF = np.zeros(s)
    Gbest = np.zeros(Nc)
    GbestF = sys.maxsize
    for i in range(s):
        PbestF[i] = sys.maxsize
    for i in range(50):
        [F, S] = calculate(data, S, M)
        [Pbest, PbestF, Gbest, GbestF] = getBestParticle(
            S, Pbest, PbestF, Gbest, GbestF, F, Nc)
        [S, V] = UpdateParticle(S, V, Pbest, Gbest, Nc)
    [Mt, a] = getMt(Gbest, M)
    return Mt


def Loop2(data, Nc):
    M = initCentroid(data, Nc)
    for i in range(5):
        Mt = Loop1(data, M, Nc)
        Mt = kmeans(Mt, data)
        M = UpdateM(data, Mt, Nc)
    return Mt


def Test1():
    data = np.loadtxt('dataset/compound/compound.txt')
    centroid = Loop2(data, 12)
    print(centroid)
    print(DBIndex(data, Assign(centroid, data), centroid))


if __name__ == "__main__":
    Test1()
