import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle

def fuzzify(R, base):
    y = np.zeros_like(R, dtype = float)
    n = np.zeros_like(R, dtype = float)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            y[i,j] = (R[i,j] - np.min(base[:,j]))/(np.max(base[:,j]) - np.min(base[:,j]))
            n[i,j] = (1 - y[i,j])/(1 + y[i,j])
    return y, n

def S(y, n):
    return (3+2*y+y**2-n-2*(n**2))*np.exp(2*y-2*n-2)/6

def T_calc(R):
    T = np.zeros(R.shape[1], dtype = float)
    y, n = fuzzify(R, R)
    for j in range(R.shape[1]):
        for i in range(R.shape[0]):
            T[j] += np.abs(S(y[i,j], n[i,j]) - S(n[i,j], y[i,j]))/R.shape[0]
    return T

def weight_train(w0, R, tol = 1e-4, max_iter = 10000, check_after = 1):
    #return history of each iteration of calculating w
    w_history = []
    T = T_calc(R)
    count = 0
    w_check = w0
    while count < max_iter:
        w = w0*T/np.sum(w0*T)
        w_history.append(w)
        count += 1
        if count % check_after == 0:
            if np.linalg.norm(w - w_check) < tol: return np.asarray(w_history, dtype = 'float')
            w_check = w
        w0 = w
    return np.asarray(w_history, dtype = 'float')

class OrderedLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

def center_find(R, L, classes):
    encoder = OrderedLabelEncoder()
    encoder.fit(classes)
    L_encoded = encoder.transform(L)
    
    R_Lp = np.zeros((classes.shape[0], R.shape[1]))
    
    for i in range(classes.shape[0]):
        count = np.argwhere(L_encoded == i).shape[0]
        R_Lp[i] = np.sum(R[np.argwhere(L_encoded == i).flatten()], axis = 0)/count
    
    return R_Lp

def distance_calc(yi, ni, y_Lp, n_Lp, w, ruler = 'eu'):
    #this function is used to calculate distance between an object and a class center
    #you can choose the measure method by changing the parameter 'ruler' (which is defaultly set to 'eu')
    #note: parameter: ruler: {
    #                         'eu': Euler's distance
    #                         'ha2': Hamming's distance with y and n function
    #                         'ha3': Hamming for y, n and h = 1 - y - n
    #                         'ng': Ngan's distance
    #                         'ma': Mahanta's distance
    #                        }, optional
    
    dist = 0
    
    if ruler == 'eu':
        Si = S(yi, ni)
        S_Lp = S(y_Lp, n_Lp)
        dist = np.sum(w*np.abs(Si - S_Lp))
    
    def abs_diff(ai, a_Lp):
        return np.abs(ai - a_Lp)
    
    y_diff = abs_diff(yi, y_Lp)
    n_diff = abs_diff(ni, n_Lp)
    
    if ruler == 'ha2':
        dist = np.sum(w*(y_diff+n_diff)/2)
    
    if ruler == 'ha3':
        hi = 1 - yi - ni
        h_Lp = 1 - y_Lp - n_Lp
        h_diff = abs_diff(hi, h_Lp)
        dist = np.sum(w*(y_diff+n_diff+h_diff)/2)
    
    if ruler == 'ng':
        max_y = np.zeros(yi.shape[0])
        max_n = np.zeros_like(max_y)
        for i in range(max_y.shape[0]):
            max_y[i] = np.max(np.array([yi[i], n_Lp[i]]))
            max_n[i] = np.max(np.array([y_Lp[i], ni[i]]))
            
        dist = np.sum(w*((y_diff+n_diff)/4+np.abs(max_y - max_n)/2)/3)
    
    if ruler == 'ma':
        dist = np.sum(w*((y_diff+n_diff)/(yi+ni+y_Lp+n_Lp)))
    
    return dist

def labeling(R, R_Lp, classes, w, ruler):
    L_pred = np.zeros(R.shape[0], dtype = classes.dtype)
    y, n = fuzzify(R, R)
    y_Lp, n_Lp = fuzzify(R_Lp, R)
    
    for i in range(R.shape[0]):
        dists = np.zeros(classes.shape[0])
        for j in range(classes.shape[0]):
            dists[j] = distance_calc(y[i], n[i], y_Lp[j], n_Lp[j], w, ruler)
        L_pred[i] = classes[np.argmin(dists)]
    
    return L_pred
            

def accuracy(L_pred, L_real):
    return np.argwhere(L_pred == L_real).shape[0]/L_real.shape[0]

def early_stopping(w_history, L, R, R_Lp, classes, ruler, momentum = 2):
    w = w_history[0]
    counter = 0
    for i in range(1, w_history.shape[0]):
        if accuracy(labeling(R, R_Lp, classes, w, ruler), L) <= accuracy(labeling(R, R_Lp, classes, w_history[i], ruler), L):
            w = w_history[i]
            counter = 0
        else:
            counter += 1
        if counter > momentum: break
    return w

def value_of_confusion_matrix(L_pred, L_real, classes):
    TP=[]
    FP=[]
    TN=[]
    FN=[]
    if(classes.dtype == 'float64'):
        for i in range(classes.shape[0]):
            TP.append(np.argwhere((classes[i].astype('str')==L_pred.astype('str'))&(classes[i].astype('str')==L_real.astype('str'))).shape[0])
            FP.append(np.argwhere((classes[i].astype('str')==L_pred.astype('str'))&(classes[i].astype('str')!=L_real.astype('str'))).shape[0])
            TN.append(np.argwhere((classes[i].astype('str')!=L_pred.astype('str'))&(classes[i].astype('str')!=L_real.astype('str'))).shape[0])
            FN.append(np.argwhere((classes[i].astype('str')!=L_pred.astype('str'))&(classes[i].astype('str')==L_real.astype('str'))).shape[0])
    else:
        for i in range(classes.shape[0]):
            TP.append(np.argwhere((classes[i]==L_pred)&(classes[i]==L_real)).shape[0])
            FP.append(np.argwhere((classes[i]==L_pred)&(classes[i]!=L_real)).shape[0])
            TN.append(np.argwhere((classes[i]!=L_pred)&(classes[i]!=L_real)).shape[0])
            FN.append(np.argwhere((classes[i]!=L_pred)&(classes[i]==L_real)).shape[0])

    return TP,FP,TN,FN

def confusion_matrix(L_pred, L_real, classes):
    TP,FP,TN,FN = value_of_confusion_matrix(L_pred, L_real, classes)
    return np.array([TP,FP,FN,TN]).T.reshape(-1, 2, 2)

def acc(L_pred, L_real, classes):
    TP, FP, TN, FN = value_of_confusion_matrix(L_pred,L_real, classes)
    ACC=[]
    for i in range(classes.shape[0]):
        ACC.append((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]))
    return ACC

def sen(L_pred, L_real, classes):
    TP, FP, TN, FN = value_of_confusion_matrix(L_pred,L_real, classes)
    SEN=[]
    for i in range(classes.shape[0]):
        SEN.append(TP[i]/(TP[i]+FN[i]))
    return SEN

def spec(L_pred, L_real, classes):
    TP, FP, TN, FN = value_of_confusion_matrix(L_pred,L_real, classes)
    SPEC=[]
    for i in range(classes.shape[0]):
        SPEC.append(TN[i]/(TN[i]+FP[i]))
    return SPEC

def pre(L_pred, L_real, classes):
    TP, FP, TN, FN = value_of_confusion_matrix(L_pred,L_real, classes)
    PRE=[]
    for i in range(classes.shape[0]):
        PRE.append(TP[i]/(TP[i]+FP[i]))
    return PRE

def f1_score(L_pred, L_real, classes):
    TP, FP, TN, FN = value_of_confusion_matrix(L_pred,L_real, classes)
    F1=[]
    for i in range(classes.shape[0]):
        F1.append(2*TP[i]/(2*TP[i]+FP[i]+FN[i]))
    return F1

def prob_label(R, R_Lp, classes, w, ruler):
    prob = np.zeros((R.shape[0], classes.shape[0]))
    y, n = fuzzify(R, R)
    y_Lp, n_Lp = fuzzify(R_Lp, R)
    
    for i in range(R.shape[0]):
        dists = np.zeros(classes.shape[0])
        for j in range(classes.shape[0]):
            dists[j] = distance_calc(y[i], n[i], y_Lp[j], n_Lp[j], w, ruler)
        prob[i] = np.exp(-dists)/np.exp(-dists).sum(axis = 0)
    
    return prob

def draw_ROC(R, R_Lp, classes, w, ruler, L_real, L_pred):
    prob = prob_label(R, R_Lp, classes, w, ruler)
    encoder = OrderedLabelEncoder()
    encoder.fit(classes)
    classes_encoded = encoder.transform(classes)
    L_real_encoded = encoder.transform(L_real)
    L_pred_encoded = encoder.transform(L_pred)
    label_binarizer = LabelBinarizer().fit(classes_encoded)
    L_onehot_test = label_binarizer.transform(L_real_encoded)
    L_onehot_pred = label_binarizer.transform(L_pred_encoded)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(classes.shape[0]):
        fpr[i], tpr[i], _ = roc_curve(L_onehot_test[:, i], prob[:, i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(classes.shape[0]):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i], left = 0)
    mean_tpr /= classes.shape[0]
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    tpr["macro"][0] = 0
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fpr["micro"], tpr["micro"], _ = roc_curve(L_onehot_test.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "lavender","red", "purple","darkgreen", "midnightblue", ])
    for i, color in zip(range(classes.shape[0]), colors):
        RocCurveDisplay.from_predictions(
            L_onehot_test[:, i],
            prob[:, i],
            name=f"ROC curve for {classes[i]}",
            color=color,
            ax=ax,
        )
    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves of all class together\nto One-vs-Rest multiclass")
    plt.legend()
    return fig