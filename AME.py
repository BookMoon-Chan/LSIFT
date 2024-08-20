#Submodel
import numpy as np
import math
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn

def Ensemble(OUT,output,La,N,omegaalpha,k,Enq,cu):
    en=torch.Tensor(np.ones((N,k))).cuda(cu)
    enn=torch.mul(omegaalpha,en)
    out=torch.mul(omegaalpha,output)
    for i in range(k):
        OUT[:,La[i]]=OUT[:,La[i]]+out[:,i]
        Enq[:,La[i]]=Enq[:,La[i]]+enn[:,i]
    return OUT,Enq

def Attention(OUT,Enq,cu):
    OUT=torch.div(OUT,Enq)
    return OUT


