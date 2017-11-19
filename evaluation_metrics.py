from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def calculate_AUC(predicted, actual):
    #Inputs are one-dimensional numpy arrays
    fpr, tpr, thresholds = metrics.roc_curve(actual.ravel(), predicted.ravel(), pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    return AUC, fpr, tpr

def calculate_AUPR(predicted, actual):
    #Inputs are one-dimensional numpy arrays
    AUPR = metrics.average_precision_score(actual.ravel(), predicted.ravel())
    return AUPR

def AUC_parameters(org):
    plt.figure(figsize=[8,8])
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    lw = 2
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.title('ROC Curves for Models: '+org, size=20)

def AUPR_parameters(org):
    plt.figure(figsize=[8,8])
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        
    lw = 2
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.title('Precision-Recall Curves: '+org, size=20)

def plot_AUC_curve(predicted, actual, label, org):
    #Label corresponds to a model type -- e.g. 'FastText'
    #org corresponds to an organism type -- e.g. 'Human'
    AUC_parameters(org)
    AUC, fpr, tpr = calculate_AUC(predicted, actual)
    plt.plot(fpr, tpr, lw=2, label=label+' (AUC = %0.2f)' % AUC)
    plt.legend(loc="lower right")

def plot_AUPR_curve(predicted, actual, label, org):
    #Label corresponds to a model type -- e.g. 'FastText'
    #org corresponds to an organism type -- e.g. 'Human'
    AUPR_parameters(org)
    AUPR = calculate_AUPR(predicted, actual)
    precision, recall,_ = metrics.precision_recall_curve(actual.ravel(), predicted.ravel(), pos_label=1)
    FScores = 2*precision*recall/(precision+recall)
    Fmax = max(FScores)
    FmaxIndex = list(FScores).index(Fmax)
    plt.plot(recall, precision, lw=2, label=label+' (AUPR = %0.2f, FMax = %0.2f)' % (AUPR, Fmax))
    plt.plot(recall[FmaxIndex], precision[FmaxIndex], '-bo', markersize=12)
    plt.legend(loc="lower right")