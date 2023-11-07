import numpy as np
import collections
import yaml
import pandas as pd
import requests
import pydicom
import cv2
import torch
import math
import os

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


import tarfile
import io


def read_tar_dicom(tar_file_path):
    tar_contents = {}
    try:
        # Open the tar file as a binary stream
        with tarfile.open(tar_file_path, "r") as tar:
            # Iterate through the files in the tar archive
            for tar_info in tar:
                # Check if the tar entry is a regular file (not a directory or a symlink)
                if tar_info.isfile():
                    # Read the content of the file into a variable
                    content = tar.extractfile(tar_info).read()

                    # Store the content in the dictionary with the file name as the key
                    tar_contents[tar_info.name] = content

    except tarfile.TarError as e:
        print(f"Error while processing the tar file: {e}")

    return tar_contents


# print(tar_contents.keys())
# print(len(tar_contents.keys()))
# Now, tar_contents contains the contents of each file in the tar archive
# You can access the content of a specific file like this:
# content_of_file = tar_contents['file_name_inside_tar']


def get_latest_ckpt(config):
    config_ckpt, dataset_target = config.ckpt, config.dataset.target
    assert os.path.isdir(config_ckpt), f"{config_ckpt} is not a directory"
    if config_ckpt.endswith(".ckpt"):  # and not os.path.isfile(config_ckpt):
        latest_ckpt = config_ckpt
    else:
        task_paths = [
            os.path.join(config_ckpt, task_path)
            for task_path in os.listdir(config_ckpt)
            if dataset_target in task_path
        ]
        ckpt_paths = [
            os.path.join(task_path, ckpt_path)
            for task_path in task_paths
            for ckpt_path in os.listdir(task_path)
            if ckpt_path.endswith(".ckpt")
        ]
        latest_ckpt = max(ckpt_paths, key=os.path.getctime)
        # while not os.path.isfile(latest_ckpt):
        #     ckpt_paths.remove(latest_ckpt)
        #     latest_ckpt = max(ckpt_paths, key=os.path.getctime)
    print(f"Loading latest checkpoint: {latest_ckpt}")
    return latest_ckpt
