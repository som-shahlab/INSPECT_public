import numpy as np
import collections
import yaml
import pandas as pd
import requests
import pydicom
import cv2
import torch
import math

from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


def get_auroc(y, prob, keys):

    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)
    if len(prob.shape) == 1:
        prob = np.expand_dims(prob, -1)

    auroc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            auroc_dict[k] = 0.0
        elif len(set(y_cls)) == 1:
            auroc_dict[k] = 0.0
        else:
            auroc_dict[k] = roc_auc_score(y_cls, prob_cls)
    auroc_dict["mean"] = np.mean([v for _, v in auroc_dict.items()])
    return auroc_dict


def get_auprc(y, prob, keys):

    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()
        
    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)
    if len(prob.shape) == 1:
        prob = np.expand_dims(prob, -1)

    auprc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            auprc_dict[k] = 0.0
        elif len(set(y_cls)) == 1:
            auprc_dict[k] = 0.0
        else:
            auprc_dict[k] = average_precision_score(y_cls, prob_cls)
    auprc_dict["mean"] = np.mean([v for _, v in auprc_dict.items()])
    return auprc_dict
