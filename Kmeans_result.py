import numpy as np
from KMeans import Kmeans, storeResult, Assign

repeat = 30
filename = "AllResult_Kmeans.txt"


def getResult(k, dataName):
    path = 'dataset/'+dataName+'/'+dataName+'.txt'
    print(path)
    data = np.loadtxt(path, dtype=float)
    f = open(filename, 'a')
    f.writelines('This is '+dataName+'\n')
    f.close()
    for i in range(repeat):
        centroid = Kmeans(k, data)
        storeResult(data, centroid, filename)


def getKmeansResult():
    filenames = ['aggregation', 'compound', 'seeds', 'abalone', 'avila', 'HTRU2',
                 '2d15c', '2d8c', '2d10c', '2d20c', '4d8c', '4d15c', '4d20c', '10d8c',
                 '10d10c', '10d15c', '10d20c', '20d10c', '20d15c']
    centrtroidNum = np.array(
        [7, 6, 3, 3, 12, 2, 15, 8, 10, 20, 8, 15, 20, 8, 10, 10, 15, 10, 15])
    for i in range(len(filenames)):
        filename = filenames[i]
        k = centrtroidNum[i]
        getResult(k, filename)
        print('finish  :'+filename)


if __name__ == '__main__':
    getKmeansResult()
