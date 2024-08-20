#Save
from scipy.io import savemat

def Save(A,name,dataname):
    savemat(dataname+'//'+name+'-'+dataname+'.mat', {name:A})
    return

def Saves(A,name,NAME,dataname):
    savemat(NAME+'//'+dataname+'-'+NAME+'.mat', {name:A})
    return



