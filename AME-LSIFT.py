import math;import random;import numpy as np;import time
from sklearn.model_selection import KFold
import scipy.io as sio
import torch
import torch.nn as nn
from torch.autograd import Variable

#function
import Labelsubset;import Centers;import LSIFT;import Submodel;import AME;import Metrics;import Save
criterion=nn.CrossEntropyLoss()#cross entropy loss
#参数设定
cu=1#0,1,2,3
print(cu)
ZHE=5;zhe=5
BATCH=256;EPOCHS=1000#500
learning_rate=0.001;LEARNING_RATE=0.001
k=3;r=0.1;lamb=0.00001;gamma=0.5
distance_type=2#1:E;2:G_kernel;
datanames=["scene","CAL500","CHD_49","chemistry","chess","coffee","emotions","enron","foodtruck","flags",
           "genbase","GramnegativeGoB1392","GrampositiveGoB519","HumanPseACC3106","languagelog","philosophy","PlantGoB978","scene","slashdot","sourcesBBC",
           "sourcesGuardian","sourcesReuters","waterquality","yeast","yelp"]
dataname=datanames[9]
print(dataname)
file = dataname + '-instances' + '.mat';load_fn = file;load_data = sio.loadmat(load_fn)
OX = load_data['Data'];#OX=OX.todense()
file = dataname + '-labels' + '.mat';load_fn = file;load_data = sio.loadmat(load_fn)
OY = load_data['data']
[ON,q]=OY.shape;d=OX.shape[1];n=2*q

ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer_ExpLR, gamma=0.99)

tims=0
BATCHS=math.floor(ON*0.8/BATCH);print(ON*0.8/BATCH)
HUALOSS = np.zeros((ZHE*zhe, EPOCHS*BATCHS))
METRICS=np.zeros((ZHE*zhe,11))#1.Hloss;2.Fexam;3.MacF;4.MicF;5.Accuracy;6.TIME
for tim in range(ZHE):
     L=Labelsubset.Labelsubset(q, k).astype(int);Lname='L'+str(tim);globals()[Lname]=L
     xu=np.arange(ON);random.shuffle(xu);kf=KFold(n_splits=zhe)#k-fold中的k
     for train, test in kf.split(xu):
         trainxu=np.array(xu)[train]
         testxu=np.array(xu)[test]
         N=train.shape[0];BATCHS=math.floor(N/BATCH)
         if BATCHS==0:
             BATCHES=1
         batchrest=N%BATCH
         NN=test.shape[0]
         xtrain = np.zeros((N,d));train_bias=np.ones((N,1))
         ytrain = np.zeros((N,q))
         xtest = np.zeros((NN,d));test_bias=np.ones((NN, 1))
         ytest = np.zeros((NN,q))
         for i in range(N):
             aim = trainxu[i];xtrain[i] = OX[aim];ytrain[i] = OY[aim]
             total_Y = torch.Tensor(ytrain).cuda(cu)
         for i in range(NN):
             aim = testxu[i];xtest[i] = OX[aim];ytest[i] = OY[aim]
         TIME=0;time_begin=time.time()
         for alpha in range(n):#LSIFT_alpha#submodel
             Cname = 'center' + str(alpha);globals()[Cname] = Centers.Centers(xtrain, ytrain, L[alpha, :], k, r)  # center_alpha
             xalpha = np.append(LSIFT.LSIFT(xtrain, globals()[Cname], distance_type), train_bias, axis=1)
             Xname = 'xtrain_LSIFT_' + str(alpha);globals()[Xname] = np.append(xtrain, xalpha, axis=1)
             d_a=globals()[Xname].shape[1]
             Mname='model'+str(alpha);model=globals()[Mname]=Submodel.BPNN(d_a,k).cuda(cu)
             xxalpha = np.append(LSIFT.LSIFT(xtest, globals()[Cname], distance_type), test_bias, axis=1)
             XXname = 'xtest_LSIFT_' + str(alpha);globals()[XXname] = np.append(xtest, xxalpha, axis=1)
         Honame='HUAOMEGA'+str(tims);globals()[Honame]=np.zeros((n,EPOCHS*BATCHS))
         Hslname='HUASUBLOSS'+str(tims);globals()[Hslname]=np.zeros((n,EPOCHS*BATCHS))
         ome = torch.Tensor(np.zeros((n))).cuda(cu)
         omega = torch.sigmoid(ome).cuda(cu)#nn.functional.softmax(ome).cuda(cu)
         grad_variables = ome.cuda(cu)
         ome = Variable(ome, requires_grad=True).cuda(cu)
         omegaoptimizer = torch.optim.Adam([ome], lr=learning_rate)
         omegaoptimizer.zero_grad()
         ci=0
         for epoch in range(EPOCHS):
             Y = total_Y.cuda(cu)
             head=0;tail=batchrest+BATCH
             for bat in range(BATCHS):
                 Honame = 'HUAOMEGA' + str(tims);globals()[Honame][:, ci] = omega.cpu().detach().numpy()
                 subloss = torch.Tensor([0.0]).cuda(cu)
                 l1 = torch.Tensor([0.0]).cuda(cu)
                 OUT = torch.Tensor(np.zeros((tail-head, q))).cuda(cu)
                 Enq = torch.Tensor(np.zeros((tail-head, q))).cuda(cu)
                 Hslname = 'HUASUBLOSS' + str(tims);Huasubloss = globals()[Hslname]
                 for alpha in range(n):
                     Xname = 'xtrain_LSIFT_' + str(alpha);
                     xt = globals()[Xname]
                     yt = ytrain[:, L[alpha, :]]
                     train_ds = Submodel.Tabulardataset(xt, yt)
                     batchs = tail-head
                     xss, yss = Submodel.Datatensor(train_ds, N, cu)
                     xs=xss[head:tail,:];ys=yss[head:tail,:]
                     Mname = 'model' + str(alpha);model = globals()[Mname]
                     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                     optimizer.zero_grad()
                     output = model(xs);output = output.cuda(cu);Oname = 'output' + str(alpha);globals()[Oname] = output
                     Huasubloss[alpha, ci] = criterion(output, ys).item()+Submodel.l1_regular(model, lamb).item()
                     subloss = subloss + criterion(output, ys);subloss.cuda(cu)
                     l1 = l1 + Submodel.l1_regular(model, lamb);l1.cuda(cu)
                     OUT, Enq= AME.Ensemble(OUT, output, L[alpha, :], tail-head, omega[alpha], k, Enq, cu)
                 Total_out = AME.Attention(OUT, Enq, cu)
                 R2 = criterion(Total_out, Y[head:tail,:]);R2.cuda(cu)
                 R1 = (subloss + l1) / n;R1.cuda(cu)
                 Loss = gamma * R1 + (1 - gamma) * R2;Loss.cuda(cu)
                 loss = Loss.item();print(loss,ci)
                 omegaoptimizer.zero_grad()
                 Loss.backward()
                 omegaoptimizer.step()
                 omega = torch.sigmoid(ome).cuda(cu)
                 for alpha in range(n):
                     Mname = 'model' + str(alpha);model = globals()[Mname]
                     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                     optimizer.step()
                     optimizer.zero_grad()
                     globals()[Mname]=model
                 HUALOSS[tims, ci] = loss
                 ci=ci+1
                 head = tail;tail = head + BATCH
         time_end = time.time();TIME = time_end - time_begin
         ########################TESTING
         OUTest = torch.Tensor(np.zeros((NN, q))).cuda(cu)
         Enqest = torch.Tensor(np.zeros((NN, q))).cuda(cu)
         for alpha in range(n):
             XXname='xtest_LSIFT_'+str(alpha);xx_a=torch.Tensor(globals()[XXname]).cuda(cu)
             Mname='model'+str(alpha);model=globals()[Mname]
             output=model(xx_a).cuda(cu)
             OUTest, Enqest= AME.Ensemble(OUTest, output, L[alpha, :], NN, omega[alpha], k, Enqest, cu)
         Total_out_test = AME.Attention(OUTest, Enqest, cu)
         METRICS=Metrics.Metrics(Total_out_test, ytest, TIME, METRICS, tims)
         Honame = 'HUAOMEGA' + str(tims);huaomega=globals()[Honame]
         Save.Save(huaomega, 'HUAOMEGA'+str(tims), dataname)
         Hsname = 'HUASUBLOSS' + str(tims);huasubloss = globals()[Hsname]
         Save.Save(huasubloss, 'HUASUBLOSS' + str(tims), dataname)
         Save.Save(L, 'Labelsubset' + str(tims), dataname)
         tims=tims+1
         print('shizhege',tims)
     print('times',tim)
     print(METRICS)

Save.Save(HUALOSS,'HUALOSS',dataname)
Save.Save(METRICS,'aMETRICS',dataname)
print(METRICS)