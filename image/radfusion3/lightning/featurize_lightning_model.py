import numpy as np
import torch
import pandas as pd
import os
import h5py 
import tqdm

from collections import defaultdict
from .. import builder
from ..constants import *
from pytorch_lightning.core import LightningModule


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

        self.features_dir = os.path.join(
            self.cfg.output_dir, 
            f"{self.cfg.model.model_name}_{self.cfg.dataset.transform.resize_size}_{self.cfg.dataset.transform.channels}_features")
        if not os.path.isdir(self.features_dir):
            os.makedirs(self.features_dir)


    def training_step(self, batch, batch_idx):
        raise Exception('Training not supported for featurize model')

    def validation_step(self, batch, batch_idx):
        raise Exception('Validation not supported for featurize model')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_training_epoch_end(self):
        raise Exception('Training not supported for featurize model')

    def on_validation_epoch_end(self):
        raise Exception('Validation not supported for featurize model')

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split, extract_features=False):
        """Similar to traning step"""

        x, y, instance_id = batch
        logit, features = self.model(x, get_features=True)

        for ids, f in zip(instance_id, features.cpu().detach().numpy()):
            try: 
                np.save(os.path.join(self.features_dir, f"{ids}.npy"), f)
            except: 
                print('[ERROR]', ids)

        return None

    def shared_epoch_end(self, step_outputs, split):
        
        df = pd.read_csv(self.cfg.dataset.csv_path)
        df['patient_datetime'] = df.apply(
            lambda x: f"{x.AnonPersonId}_{x.AnonProcedureDatetime}",
            axis=1
        )

        hdf5_path = os.path.join(
            self.features_dir,
            f"{self.cfg.model.model_name}_{self.cfg.dataset.transform.resize_size}_{self.cfg.dataset.transform.channels}_features.hdf5")
        hdf5_fn = h5py.File(hdf5_path, 'w')

        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)): 
            pdt = row['patient_datetime']
            features = []
            for instance_idx in range(row.num_slices):
                try:
                    features.append(np.load(os.path.join(self.features_dir, f"{pdt}@{instance_idx}.npy")))
                except:
                    print(os.path.join(self.features_dir, f"{pdt}@{instance_idx}.npy"))
            if len(features) == 0:
                print(os.path.join(self.features_dir, f"{pdt}.npy"))
                continue
            features = np.stack(features)
            hdf5_fn.create_dataset(pdt, data=features, dtype='float32', chunks=True)
        hdf5_fn.close()
        print(f"\nFeatures saved at: {hdf5_path}")
