# Compute ROC curve and ROC area 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def evaluate(y_test, y_pred, y_proba):
    """
    This function compute the performance's metrics for one model
    
    Args:
    -----
    y_test (pd.Series): True labels
    y_pred (pd.Series): Predicted labels
    y_proba (array): Probability predicted
    
    Return:
    -------
    dict_metrics (dict): dictionary that contains the recall, precision,
                         accuracy, f1-score and AUC
    """
    
    vp = np.sum((y_test == 1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    vn = np.sum((y_test == 0) & (y_pred == 0))
    
    # calcular el recall
    recall = vp / (vp + fn)
    
    # calcular el precision  
    precision = vp / (vp + fp)
    
    # accuracy
    acc = (vp + vn) / (vp + fn + fp + vn)
    
    # f1-score
    f1 = 2 * precision * recall / (precision + recall)
    
    # AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    dict_metrics = {'recall' : recall, 
                    'precision' : precision,
                    'f1' : f1,
                    'accuracy' : acc,
                    'auc' : roc_auc}
    
    # graficas
    # AUC
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return dict_metrics