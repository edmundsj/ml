from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve
from matplotlib import pyplot as plt
from sklearn.base import clone

def plot_precision_recall(actual_classes, scores, method='threshold', fig=None, ax=None):
    """
    :param method: Options are "threshold", "together", "ROC", "all"
    """
    if fig is None:
        fig, ax = plt.subplots()
    elif method == 'threshold':
        precisions, recalls, thresholds = precision_recall_curve(actual_classes, scores)
        ax.plot(thresholds, precisions[:-1], label='precision')
        ax.plot(thresholds, recalls[:-1], label='recall')
        ax.legend(['precision', 'recall'])
    elif method == 'together':
        precisions, recalls, thresholds = precision_recall_curve(actual_classes, scores)
        ax.plot(recalls, precisions)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
    elif method == 'ROC':
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual_classes, scores)
        ax.plot(false_positive_rate, true_positive_rate)
        ax.plot([0, 1], [0, 1], linestyle='dashed')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (recall)')

    return fig, ax


