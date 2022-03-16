import os
from skimage import measure
from sklearn.metrics import roc_curve, auc
import numpy as np


def evaluate(labels, scores, metric='roc'):
    if metric == 'pro':
        return pro(labels, scores)
    if metric == 'roc':
        return roc(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")


def roc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def pro(masks, scores):
    '''
        https://github.com/YoungGod/DFR/blob/a942f344570db91bc7feefc6da31825cf15ba3f9/DFR-source/anoseg_dfr.py#L447
    '''
    # per region overlap
    max_step = 4000
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[i])
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    expect_fpr = 0.3
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr    # # rescale fpr [0, 0.3] -> [0, 1]
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])    # need scale
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score
