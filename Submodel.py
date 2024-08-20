#Submodel
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn

class BPNN(nn.Module):
    def __init__(self,d,q):
        super().__init__()
        self.lin1 = nn.Linear(d, 1024)
        self.lin2 = nn.Linear(1024, 128)
        self.lin3 = nn.Linear(128, q)
    def forward(self, x_in):
        x = F.relu(self.lin1(x_in))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x

class Tabulardataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def Datatensor(train_ds,batch_size,cu):
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    for i, (xs, ys) in enumerate(train):
        xs = xs.float().cuda(cu)
        ys = ys.float().cuda(cu)
    return xs,ys

def l1_regular(model,lamb):
    l1_loss=[]
    for name,parameters in model.state_dict().items():
        if "weight" in name:
            l1_loss.append(torch.abs(parameters).sum())
            l1=lamb*sum(l1_loss)
    return l1

