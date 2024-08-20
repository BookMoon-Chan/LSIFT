#Centers
import numpy as np
import math
from sklearn.cluster import KMeans
def Centers(X,Y,La,k,r):
    Y=Y[:,La]
    [N,d]=X.shape
    Ta=np.zeros((N,1)).astype(int)
    group=2**k
    G=np.zeros((1,group))
    centers=np.zeros((0,d))
    for g in range(group):
        nameG='G'+str(g)
        globals()[nameG]=np.zeros((0,d))
    for i in range(N):
        for j in range(k):
            Ta[i,0]=int(Ta[i,0]+Y[i,j]*(2**j))
        G[0, Ta[i, 0]] = G[0, Ta[i, 0]] + 1
        for g in range(group):
            nameG='G'+str(Ta[i,0])
            GG=globals()[nameG]
            globals()[nameG]=np.vstack((GG,X[i,:]))
    for g in range(group):
        c=math.ceil(r*G[0,g])
        if c>0:
            c_means=KMeans(n_clusters=c)
            nameG='G'+str(g)
            center=c_means.fit(globals()[nameG]).cluster_centers_
            centers=np.vstack((centers,center))
    return centers

def Center(X,Y,La,k,r):
    Y[:,0]=Y[:,La]
    [N,d]=X.shape
    Ta=np.zeros((N,1)).astype(int)
    group=2**k
    G=np.zeros((1,group))
    centers=np.zeros((0,d))
    for g in range(group):
        nameG='G'+str(g)
        globals()[nameG]=np.zeros((0,d))
    for i in range(N):
        for j in range(k):
            Ta[i,0]=int(Ta[i,0]+Y[i,j]*(2**j))
        G[0, Ta[i, 0]] = G[0, Ta[i, 0]] + 1
        for g in range(group):
            nameG='G'+str(Ta[i,0])
            GG=globals()[nameG]
            globals()[nameG]=np.vstack((GG,X[i,:]))
    for g in range(group):
        c=math.ceil(r*G[0,g])
        if c>0:
            c_means=KMeans(n_clusters=c)
            nameG='G'+str(g)
            center=c_means.fit(globals()[nameG]).cluster_centers_
            centers=np.vstack((centers,center))
    return centers




