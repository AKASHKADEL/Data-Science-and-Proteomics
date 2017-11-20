from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def calculate_AUC(predicted, actual):
    #Inputs are 2-D numpy arrays
    fpr, tpr, thresholds = metrics.roc_curve(actual.ravel(), predicted.ravel(), pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    return AUC, fpr, tpr

def calculate_AUPR(predicted, actual):
    #Inputs are 2-D numpy arrays
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


def find_GoTerm_aucs(predicted, actual):
    '''
    This function generates a list of ROC_AUC values.
    The list has one AUC value for each Go Term
    '''
    list_of_aucs = []
    for col in range(predicted.shape[1]):
        pred = predicted[:,col]
        act = actual[:,col]
        column_auc = metrics.roc_auc_score(act, pred)
        list_of_aucs.append(column_auc)
    return list_of_aucs

def find_GoTerm_auprs(predicted, actual):
    '''
    This function generates a list of AUPR values.
    The list has one AUPR value for each Go Term
    '''
    list_of_auprs = []
    for col in range(predicted.shape[1]):
        pred = predicted[:,col]
        act = actual[:,col]
        column_aupr = metrics.average_precision_score(act, pred)
        list_of_auprs.append(column_aupr)
    return list_of_auprs

def plot_GoTerm_Bars(score_list, label, org, metric = 'AUC'):
    '''
    score_list is a list of AUCs, one AUC for each GoTerm. Length = # of GoTerms
    @label corresponds to a model type -- e.g. 'FastText'
    @org corresponds to an organism type -- e.g. 'Human'
    @metric should be "AUC" or "AUPR"
    '''
    plt.figure(figsize=[10,8])
    positions = np.arange(len(score_list))
    mean_score = np.mean(score_list)
    scores_sorted = np.flipud(np.sort(score_list))
    colors = np.where(scores_sorted>0.5, 'blue', 'red')

    barlist = plt.bar(positions, scores_sorted-0.5, align='center', alpha=0.6, width=1, edgecolor='blue')

    for i in range(len(positions)):
        if colors[i] == 'red':
            barlist[i].set_color('salmon')
        
    plt.axhline(y=mean_score-0.5, color='navy', linestyle=':')
    plt.axhline(y=0, color='k', linestyle='-')
    plt.tick_params(axis='x',bottom='off',labelbottom='off')

    plt.annotate('Random Classifier', xy=(8/10*len(score_list), 0.01))
    plt.annotate('Macro '+metric, xy=(8/10*len(score_list), mean_score-0.49))
    plt.ylabel(metric, size=14)
    plt.title(metric+'s of Go Terms: '+label+' - '+org, size=16)
    plt.yticks(np.arange(-0.5, 0.6, 0.1), np.arange(0, 1.1, 0.1))
    plt.show()

