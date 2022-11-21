###################################################################
#                          Assignment- 2                          #
#                            Group- 21                            #
#                   Sharannya Ghosh (20CS10054)                   #
#                     Aritra Mitra  (20CS30006)                   #
#                            Problem- 2                           #
###################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import math
import random
def preprocess(data):
    return pd.DataFrame({'sepal length': data[0], 'sepal width': data[1], 'petal length': data[2], 'petal width': data[3], 'class': data[4]})

def distance(x1, y1, x2, y2):  # computes euclidean distance between two points
    xdiff = x1-x2
    ydiff = y1-y2
    sqdist = (xdiff**2) + (ydiff**2)
    return math.sqrt(sqdist)


def findcenter(df, ind, kcenters):  # finds the nearest k-center to a given point
    mindist = np.inf
    center = -1
    for i in range(0, len(kcenters)):
        dist = distance(df.loc[ind]['pca_feature1'], df.loc[ind]
                        ['pca_feature2'], kcenters[i][0], kcenters[i][1])
        if(dist < mindist):
            mindist = dist
            center = i
    return center


def findmean(df, indices):  # computes new centre for a given cluster
    point = []
    x = []
    y = []
    for i in indices:
        x.append(df.loc[i]['pca_feature1'])
        y.append(df.loc[i]['pca_feature2'])
    point.append(sum(x)/len(x))
    point.append(sum(y)/len(y))
    return point


def kmeans(df, k):  # performs k-means
    randindices = random.sample(range(0, df.shape[0]), k)
    kcenters = []  # list of centers updated at each iteration
    clusters = []  # list of points associated with each cluster
    for i in range(0, k):
        centerpoint = []
        centerpoint.append(df.loc[randindices[i]]['pca_feature1'])
        centerpoint.append(df.loc[randindices[i]]['pca_feature2'])
        kcenters.append(centerpoint)
    while(True):
        clusters = []
        for i in range(0, k):
            point = []
            clusters.append(point)
        for i in range(0, df.shape[0]):
            ind = findcenter(df, i, kcenters)
            clusters[ind].append(i)
        flag = True
        newcenters = []
        for i in range(0, k):
            if(len(clusters[i]) > 0):
                newcenters.append(findmean(df, clusters[i]))
            else:
                newcenters.append(kcenters[i])
            if(newcenters[i][0] != kcenters[i][0] or newcenters[i][1] != kcenters[i][1]):
                flag = False
        if(flag):
            return (kcenters, clusters)
        kcenters = newcenters.copy()


def unique(a):  # returns unique items from a given list, i.e. number of classes
    uniquelist = []
    for i in a:
        if not (i in uniquelist):
            uniquelist.append(i)
    return uniquelist


def probability(x, i):  # returns probability of item i in list x
    if(len(x) == 0):
        return 0.0
    count = 0
    for y in x:
        if(y == i):
            count = count+1
    return (count/len(x))


def entropy(x):  # returns entropy of a given set x
    classnum = unique(x)
    ent = 0.0
    for i in classnum:
        prob = probability(x, i)
        if(prob > 0):
            ent = ent+(-math.log(prob, 2)*prob)
    return ent


def mutual_info(x, y):  # returns mutual information between x and y
    ynum = unique(y)
    mutinfo = 0.0
    for i in ynum:
        proby = probability(y, i)
        xgiveny = []
        for j in range(0, len(y)):
            if(y[j] == i):
                xgiveny.append(x[j])
        ent = entropy(xgiveny)
        mutinfo = mutinfo+(proby*ent)
    mutinfo = entropy(x)-mutinfo
    return mutinfo


def NMIscore(x, y):  # computes the normalized mutual information score
    numerator = 2*mutual_info(x, y)
    denominator = entropy(x)+entropy(y)
    return(numerator/denominator)


def main():
    classmap = {  # maps each class to a particular integer
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data = pd.read_csv('iris.data',header=None)
    data = preprocess(data)
    df = data.drop('class', axis=1)
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        df[feature] = (df[feature]-df[feature].mean())/df[feature].std()
    pca = PCA(0.95)
    pca.fit(df)
    X = pca.transform(df)
    # new dataframe with modified features
    newdf = pd.DataFrame(X, columns=['pca_feature1', 'pca_feature2'])
    classlabel = []
    for label in data['class']:
        classlabel.append(classmap[label])
    plt.scatter(newdf['pca_feature1'], newdf['pca_feature2'])
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.title('Data points obtained from PCA')
    plt.show()
    newdf['classnumber'] = classlabel
    x = []
    y = []
    maxnmi = -np.inf
    maxk = 0
    for k in range(2, 9):  # k-means for k=2 to 8
        x.append(k)
        (kcenters, clusters) = kmeans(newdf, k)
        print("K-centers for k=", k)
        print(kcenters)
        cluster = [None]*(newdf.shape[0])
        label = []
        for i in range(0, k):
            for ind in clusters[i]:
                cluster[ind] = i
        for i in range(0, newdf.shape[0]):
            label.append(newdf.loc[i]['classnumber'])
        nmi = NMIscore(label, cluster)
        if(maxnmi < nmi):
            maxnmi = nmi
            maxk = k
        y.append(nmi)
    print("Maximum NMI is for k= ", maxk)
    print("Maximum NMI=", maxnmi)
    plt.plot(x, y)
    plt.xlabel('number of clusters')
    plt.ylabel('Normalized Mutual Information')
    plt.title('NMI vs k')
    plt.show()


if __name__ == "__main__":
    main()
