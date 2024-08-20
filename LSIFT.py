#LSIFT
import numpy as np
import math
def LSIFT(X,center,distance_type):
    da=center.shape[0]
    N=X.shape[0]
    Xa=np.zeros((N,da))
    Xaa = np.zeros((N, da))
    for i in range(N):
        for j in range(da):
            Xa[i,j]=np.sqrt(sum(np.power((X[i,:]-center[j,:]),2)))
            Xaa[i,j]=math.exp(-Xa[i,j]/10)
    if distance_type==2:
        Xa=Xaa
    return Xa





