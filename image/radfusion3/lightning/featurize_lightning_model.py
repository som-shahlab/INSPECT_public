import numpy as np
import pandas as pd
import os
import h5py
import tqdm

from pathlib import Path
from collections import defaultdict
from .. import builder
from ..constants import *
from pytorch_lightning.core import LightningModule
from ..utils import read_tar_dicom


class FeaturizeLightningModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.model = builder.build_model(cfg)
        self.loss = builder.build_loss(cfg)
        self.test_step_outputs = defaultdict(list)

        self.features_dir = self.cfg.dataset.output_dir
        if not os.path.isdir(self.features_dir):
            os.makedirs(self.features_dir)

    def training_step(self, batch, batch_idx):
        raise Exception("Training not supported for featurize model")

    def validation_step(self, batch, batch_idx):
        raise Exception("Validation not supported for featurize model")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_training_epoch_end(self):
        raise Exception("Training not supported for featurize model")

    def on_validation_epoch_end(self):
        raise Exception("Validation not supported for featurize model")

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        if "rsna" not in self.cfg.dataset.csv_path:
            return self.shared_epoch_end(test_step_outputs, "test")
        else:
            return self.shared_epoch_end_rsna(test_step_outputs, "test")

    def shared_step(self, batch, split, extract_features=False):
        """Similar to traning step"""
        try:
            x, y, instance_id = batch
        except:
            x, y, _, instance_id = batch
        logit, features = self.model(x, get_features=True)

        for ids, f in zip(instance_id, features.cpu().detach().numpy()):
            try:
                np.save(os.path.join(self.features_dir, f"{ids}.npy"), f)
            except:
                continue
                print("[ERROR]", ids)

        return None

    def shared_epoch_end(self, step_outputs, split):
        df = pd.read_csv(self.cfg.dataset.csv_path)

        # match dicom datetime format
        # self.df['procedure_time'] = self.df['procedure_time'].apply(
        #     lambda x: x.replace('T', ' ')
        # )
        df["procedure_time"] = df["procedure_time"].apply(lambda x: x.replace("T", " "))

        # get unique study id  by combining patient id and datetime
        # self.df["patient_datetime"] = self.df.apply(
        #     lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
        # )
        df["patient_datetime"] = df.apply(
            lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
        )

        # duplicate patient_datetime remove
        df = df.drop_duplicates(subset=["patient_datetime"])

        hdf5_path = os.path.join(self.features_dir, "features.hdf5")
        hdf5_fn = h5py.File(hdf5_path, "w")

        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            pdt = row["patient_datetime"]

            study_path = (
                Path(self.cfg.dataset.dicom_dir)
                / str(row["patient_id"])
                / str(row["procedure_time"])
            )

            # instance_path = study_path.glob("*.dcm")
            tar_content = read_tar_dicom(
                os.path.join(
                    self.cfg.dataset.dicom_dir, str(row["patient_id"]) + ".tar"
                )
            )
            prefix = "./" + str(row["procedure_time"]) + "/"
            instance_path = [
                slice_path
                for slice_path in tar_content.keys()
                if slice_path.startswith(prefix)
            ]

            slice_features = []
            # store slice features in ascending order
            for instance_idx in range(len(list(instance_path))):
                slice_feature_path = os.path.join(
                    self.features_dir, f"{pdt}@{pdt}_{instance_idx}.npy"
                )
                try:
                    slice_features.append(np.load(slice_feature_path))
                except:
                    print(slice_feature_path)

            # if len(features) == 0:
            if len(slice_features) == 0:
                print("Missing features for", pdt)
                continue
            # features = np.stack(features)
            features = np.stack(slice_features)
            hdf5_fn.create_dataset(pdt, data=features, dtype="float32", chunks=True)
        hdf5_fn.close()
        print(f"\nFeatures saved at: {hdf5_path}")

    def shared_epoch_end_rsna(self, step_outputs, split):
        df = pd.read_csv(self.cfg.dataset.csv_path)

        # match dicom datetime format
        # self.df['procedure_time'] = self.df['procedure_time'].apply(
        #     lambda x: x.replace('T', ' ')
        # )
        # df[SERIES_COL] = df[SERIES_COL]

        # get unique study id  by combining patient id and datetime
        # self.df["patient_datetime"] = self.df.apply(
        #     lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
        # )
        # df["patient_datetime"] = df.apply(
        #     lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
        # )

        # duplicate patient_datetime remove
        # df = df.drop_duplicates(subset=["patient_datetime"])

        hdf5_path = os.path.join(self.features_dir, "features.hdf5")
        hdf5_fn = h5py.File(hdf5_path, "w")

        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            pdt = row[SERIES_COL]

            study_path = (
                Path(self.cfg.dataset.dicom_dir)
                / str(row[STUDY_COL])
                / str(row[SERIES_COL])
            )

            instance_path = study_path.glob("*.dcm")
            # tar_content = read_tar_dicom(
            #     os.path.join(
            #         self.cfg.dataset.dicom_dir, str(row["patient_id"]) + ".tar"
            #     )
            # )
            # prefix = "./" + str(row["procedure_time"]) + "/"
            # instance_path = [
            #     slice_path
            #     for slice_path in tar_content.keys()
            #     if slice_path.startswith(prefix)
            # ]

            slice_features = []
            # store slice features in ascending order
            for instance_idx in range(len(list(instance_path))):
                slice_feature_path = os.path.join(
                    self.features_dir, f"{pdt}@{pdt}_{instance_idx}.npy"
                )
                try:
                    slice_features.append(np.load(slice_feature_path))
                except:
                    print(slice_feature_path)

            # if len(features) == 0:
            if len(slice_features) == 0:
                print("Missing features for", pdt)
                continue
            # features = np.stack(features)
            features = np.stack(slice_features)
            hdf5_fn.create_dataset(pdt, data=features, dtype="float32", chunks=True)
        hdf5_fn.close()
        print(f"\nFeatures saved at: {hdf5_path}")
