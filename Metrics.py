#Metrics
import numpy as np;from numpy import mat
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss

def Metrics(results,ytest,TIME,M,tims):
    yp = results.cpu().detach().numpy();y_pred = np.around(yp, 0).astype(int)  # .around()是四舍五入
    Mename='metric'+str(0);globals()[Mename]=hloss = hamming_loss(ytest, y_pred)
    Mename='metric'+str(1);globals()[Mename]=Fexam = f1_score(ytest, y_pred, average="samples")
    Mename='metric'+str(2);globals()[Mename]=MacF = f1_score(ytest, y_pred, average="macro")
    Mename='metric'+str(3);globals()[Mename]=MicF = f1_score(ytest, y_pred, average="micro")
    Mename='metric'+str(4);globals()[Mename]=zoloss = zero_one_loss(ytest, y_pred)
    Mename='metric'+str(5);globals()[Mename]=cerror = coverage_error(ytest, y_pred)
    Mename='metric'+str(6);globals()[Mename]=lrloss = label_ranking_loss(ytest, y_pred)
    Mename='metric'+str(7);globals()[Mename]=apscore = average_precision_score(ytest, y_pred)
    a1 = np.reshape(ytest, newshape=(-1));a2 = mat(y_pred);a3 = a2.A;a4 = np.reshape(a3, newshape=(-1))
    fpr, tpr, thresholds = roc_curve(a1, a4);precision, recall, _ = precision_recall_curve(a1, a4)
    Mename='metric'+str(8);globals()[Mename]=AUCROC = auc(fpr, tpr)
    Mename='metric'+str(9);globals()[Mename]=AUCPR = average_precision_score(a1, a4)
    Mename='metric'+str(10);globals()[Mename]=TIME
    for i in range(11):
        Mename='metric'+str(i)
        M[tims,i]=globals()[Mename]
    return M


def Metric(results,ytest,TIME,M,tims):
    y_pred = np.around(results, 0).astype(int)  # .around()是四舍五入
    Mename='metric'+str(0);globals()[Mename]=hloss = hamming_loss(ytest, y_pred)
    Mename='metric'+str(1);globals()[Mename]=Fexam = f1_score(ytest, y_pred, average="samples")
    Mename='metric'+str(2);globals()[Mename]=MacF = f1_score(ytest, y_pred, average="macro")
    Mename='metric'+str(3);globals()[Mename]=MicF = f1_score(ytest, y_pred, average="micro")
    Mename='metric'+str(4);globals()[Mename]=zoloss = zero_one_loss(ytest, y_pred)
    Mename='metric'+str(5);globals()[Mename]=cerror = coverage_error(ytest, y_pred)
    Mename='metric'+str(6);globals()[Mename]=lrloss = label_ranking_loss(ytest, y_pred)
    Mename='metric'+str(7);globals()[Mename]=apscore = average_precision_score(ytest, y_pred)
    a1 = np.reshape(ytest, newshape=(-1));a2 = mat(y_pred);a3 = a2.A;a4 = np.reshape(a3, newshape=(-1))
    fpr, tpr, thresholds = roc_curve(a1, a4);precision, recall, _ = precision_recall_curve(a1, a4)
    Mename='metric'+str(8);globals()[Mename]=AUCROC = auc(fpr, tpr)
    Mename='metric'+str(9);globals()[Mename]=AUCPR = average_precision_score(a1, a4)
    Mename='metric'+str(10);globals()[Mename]=TIME
    for i in range(11):
        Mename='metric'+str(i)
        M[tims,i]=globals()[Mename]
    return M


