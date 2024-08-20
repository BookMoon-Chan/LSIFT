#Labelsubset
import random
import numpy as np
def Labelsubset(q,k):
    a=0
    Laa=np.zeros((1,q))
    LA=np.zeros((0,k))
    orow=0
    while a<(2*q):
        list = range(0, q)
        La = random.sample(list,k)
        La.sort()
        La=np.mat(La)
        Laa[0,La] = 1
        num = q-Laa.sum(axis=1).item()
        Num=2*q-a-1
        if num<=Num:
            LA = np.vstack((LA, La))
            LA = np.unique(LA,axis=0)
            [row,col]=LA.shape
            if row-orow==1:
                a=a+1
                orow=row
    return LA





