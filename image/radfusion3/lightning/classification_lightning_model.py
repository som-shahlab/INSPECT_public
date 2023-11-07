import numpy as np
import torch
import torch.nn.functional as F
import wandb
import json
import pandas as pd
import pickle
import os
import h5py
import pickle

from .. import builder
from .. import utils
from ..constants import *
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.core import LightningModule
from collections import defaultdict


class ClassificationLightningModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.model = builder.build_model(cfg)
        self.loss = builder.build_loss(cfg)
        self.target_names = [""]
        self.step_outputs = defaultdict(lambda: defaultdict(list))
        self.save_dir = "./outputs"
        self.not_test_cases = []

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        # scheduler = builder.build_scheduler(self.cfg, optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    # def on_training_epoch_end(self):
    #    return self.shared_epoch_end("train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def shared_step(self, batch, split, extract_features=False):
        """Similar to traning step"""

        # x, y, instance_id, _ = batch
        x, y, mask, ids = batch
        logit, features = self.model(x, mask=mask, get_features=True)

        loss = self.loss(logit, y)

        self.log(
            f"{split}/loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        self.step_outputs[split]["logit"].append(logit)
        self.step_outputs[split]["y"].append(y)
        self.step_outputs[split]["ids"].append(ids)

        if split in ["train", "val"]:
            for i in ids:
                self.not_test_cases.append(i)

        return loss

    def shared_epoch_end(self, split):
        y = torch.cat([f for x in self.step_outputs[split]["y"] for f in x])
        logit = torch.cat([f for x in self.step_outputs[split]["logit"] for f in x])
        prob = torch.sigmoid(logit)

        if split == "test":
            config_out_dir = os.path.join(self.save_dir, "config.pkl")
            pickle.dump(self.cfg, open(config_out_dir, "wb"))

            out_dir = os.path.join(self.save_dir, "test_preds.csv")
            all_p = prob.cpu().detach().tolist()
            all_label = y.cpu().detach().tolist()
            all_ids = [f for x in self.step_outputs[split]["ids"] for f in x]
            outfile = defaultdict(list)
            for ids, label, p in zip(all_ids, all_label, all_p):
                if "rsna" not in self.cfg.dataset.csv_path:
                    pid, datetime = ids.split("_")
                elif "rsna" in self.cfg.dataset.csv_path:
                    pid = ids
                    datetime = pid
                outfile["patient_id"].append(pid)
                outfile["procedure_time"].append(datetime)
                outfile["label"].append(label)
                outfile["prob"].append(p)

            df = pd.DataFrame.from_dict(outfile)
            df.to_csv(out_dir, index=False)
            print("=" * 80)
            print(f"Config saved at: {out_dir})")
            print(f"Predictions saved at: {out_dir})")
            print("=" * 80)

        # log auroc
        auroc_dict = utils.get_auroc(y, prob, self.target_names)
        for k, v in auroc_dict.items():
            self.log(f"{split}/{k}_auroc", v, on_epoch=True, logger=True, prog_bar=True)

        # log auprc
        auprc_dict = utils.get_auprc(y, prob, self.target_names)
        for k, v in auprc_dict.items():
            self.log(f"{split}/{k}_auprc", v, on_epoch=True, logger=True, prog_bar=True)

        # self.step_outputs = defaultdict(lambda: defaultdict(list))
        del self.step_outputs[split]
