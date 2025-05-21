import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

def AUPRCutoff(y_true, y_prob):
    """
    计算 AUPRC (Area Under Precision-Recall Curve) 和最佳截断点。

    Args:
    - y_true: 真实标签，列表或数组。
    - y_prob: 模型预测的概率值，列表或数组。

    Returns:
    - best_threshold: 精确率和召回率的最佳截断点。
    - auprc: 精确率-召回率曲线下的面积。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auprc = (precision * recall).sum() / len(recall)  # 简单近似计算 AUPRC
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[f1_scores[:-1].argmax()]  # 找到最佳阈值

    return best_threshold, auprc


def sensitivity(y_true, y_prob, thresh=0.5):
    """
    计算灵敏度（召回率，真正例率）
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_prob, thresh=0.5):
    """
    计算特异性（真负例率）
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)


def auc(y_true, y_prob):
    """
    计算 AUC 值
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.roc_auc_score(y_true, y_prob)


def mcc(y_true, y_prob, thresh=0.5):
    """
    计算 Matthews 相关系数（MCC）
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_prob)


def accuracy(y_true, y_prob, thresh=0.5):
    """
    计算准确率
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_prob)


def cutoff(y_true, y_prob):
    """
    根据 ROC 和 PR 曲线计算最佳截断点
    """
    fpr, tpr, thresholds_1 = metrics.roc_curve(y_true, y_prob, drop_intermediate=False)
    precision, recall, thresholds_2 = metrics.precision_recall_curve(y_true, y_prob)

    best_roc_cutoff = thresholds_1[np.argmax(tpr - fpr)]
    best_pr_cutoff = thresholds_2[np.argmax((2 * precision * recall) / (precision + recall + 1e-8))]
    return best_roc_cutoff, best_pr_cutoff


def precision(y_true, y_prob, thresh=0.5):
    """
    计算精确率
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    return metrics.precision_score(y_true, y_prob)


def recall(y_true, y_prob, thresh=0.5):
    """
    计算召回率
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    return metrics.recall_score(y_true, y_prob)


def f1(y_true, y_prob, thresh=0.5):
    """
    计算 F1 分数
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    return metrics.f1_score(y_true, y_prob)


def AUPRC(y_true, y_prob):
    """
    计算平均精确率（AUPRC）
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)


def cofusion_matrix(y_true, y_prob, thresh=0.5):
    """
    返回混淆矩阵的元素（TN, FP, FN, TP）
    """
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) >= thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    return tn, fp, fn, tp
