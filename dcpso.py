import numpy as np
import random
import sys
from KMeans import kmeans
s = 30
c1 = 1.49
c2 = 1.49
w = 0.82
MaxV = 255
maxIteLoop1 = 30
maxTieLoop2 = 3


def Distance(d1, d2):
    sum = 0.
    for i in range(len(d1)):
        sum += (d1[i]-d2[i])*(d1[i]-d2[i])
    return np.sqrt(sum)


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


def Fitness(Z, distribute, M, Np):
    intra = 0.
    inter = 0.
    sum = 0.
    l = len(M)
    for i in range(Np):
        sum += Distance(Z[i], M[distribute[i]])
    intra = sum/Np
    inter = 0.
    for i in range(l-1):
        for j in range(i+1, l):
            if inter < Distance(M[i], M[j]) and Distance(M[i], M[j]) != 0:
                inter = Distance(M[i], M[j])
    return ((9*np.random.normal(2, 1)+1))*(intra/inter)


def Init(Nc):
    S = np.zeros((s, Nc))
    V = np.zeros((s, Nc))
    Pini = 0.8
    for i in range(s):
        for j in range(Nc):
            rk = random.randint(1, 99)/100.
            if rk < Pini:
                S[i][j] = 1
            else:
                S[i][j] = 0
            V[i][j] = (random.randint(0, 100)/10.)-5
    return S, V


def UpdataParticle(p, v, Pbest, Gbest, Nc):
    for i in range(s):
        for j in range(Nc):
            r1 = random.randint(1, 99)/100.
            r2 = random.randint(1, 99)/100.
            v[i][j] += w*v[i][j]+c1*r1 * \
                (Pbest[i][j]-p[i][j])+c2*r2*(Gbest[j]-p[i][j])
            if v[i][j] > MaxV:
                v[i][j] = MaxV
            if v[i][j] < -MaxV:
                v[i][j] = -MaxV
            sig = 1/(1+1/(np.exp(v[i][j])))
            Rk = random.randint(1, 99)/100.
            if Rk >= sig:
                p[i][j] = 0
            else:
                p[i][j] = 1
    return p, v


def Clustering(Z, Mt, Np):
    l = len(Mt)
    distribute = np.zeros(Np, dtype=int)
    for i in range(Np):
        min = sys.maxsize
        for j in range(l):
            dis = Distance(Mt[j], Z[i])
            if min > dis:
                min = dis
                distribute[i] = j
    return distribute


def GenerateM(Z, num, Nd, Np, Nc):
    M = np.zeros((num, Nd))
    t = np.zeros(num, dtype=int)
    i = 0
    while i < num:
        a = random.randint(0, Np-1)
        sum = 0
        for j in range(i):
            if a != t[j]:
                sum += 1
        if sum == i:
            t[i] = a
            i += 1
    for i in range(num):
        M[i] = Z[t[i]]
    return M


def GetMt(S, M, Nd, Nc):
    sum = 0
    for i in range(Nc):
        if S[i] == 1:
            sum += 1
    if sum < 2:
        sum = 2
    Mt = np.zeros((sum, Nd))
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


def UpdateM(Z, Mt, Nd, Np, Nc):
    if len(Mt) != Nc:
        t = Nc-len(Mt)
        a = GenerateM(Z, t, Nd, Np, Nc)
        M = np.zeros((Nc, Nd))
        for i in range(len(Mt)):
            M[i] = Mt[i]
        for i in range(len(Mt), Nc):
            M[i] = a[i-len(Mt)]
        return M
    else:
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


def Calculate(Z, S, M, Nd, Np, Nc):
    F = np.zeros(s)
    for i in range(s):
        [Mt, P] = GetMt(S[i], M, Nd, Nc)
        if len(Mt) > 2:
            Mt = testRepeat(Mt)
        d = Clustering(Z, Mt, Np)
        F[i] = Fitness(Z, d, Mt, Np)
        S[i] = P
    return F, S


def GetBestParticle(S, Pbest, PbestF, Gbest, GbestF, F, Nc):
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


def Loop1(Z, M, Nd, Np, Nc):
    [S, V] = Init(Nc)
    Pbest = np.zeros((s, Nc))
    PbestF = np.zeros(s)
    Gbest = np.zeros(Nc)
    GbestF = sys.maxsize
    for i in range(s):
        PbestF[i] = sys.maxsize
    for i in range(maxIteLoop1):
        [F, S] = Calculate(Z, S, M, Nd, Np, Nc)
        [Pbest, PbestF, Gbest, GbestF] = GetBestParticle(
            S, Pbest, PbestF, Gbest, GbestF, F, Nc)
        [S, V] = UpdataParticle(S, V, Pbest, Gbest, Nc)
    [Mt, a] = GetMt(Gbest, M, Nd, Nc)
    return Mt


def Loop2(Z, Nd, Np, Nc):
    M = GenerateM(Z, Nc, Nd, Np, Nc)
    for i in range(maxTieLoop2):
        Mt = Loop1(Z, M, Nd, Np, Nc)
        Mt = kmeans(Z, Mt)
        M = UpdateM(Z, Mt, Nd, Np, Nc)
    return Mt


def dcpso(k, data):
    Nc = k*4
    Np = len(data)
    Nd = len(data[0])
    return Loop2(data, Nd, Np, Nc)