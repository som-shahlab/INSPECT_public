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

        # Load metadata from config path
        if hasattr(cfg.dataset, 'metadata_path'):
            self.df_metadata = pd.read_csv(cfg.dataset.metadata_path)
        else:
            print("Warning: metadata_path not configured, image_id mapping will not be available")

    def training_step(self, batch, batch_idx):
        raise Exception("Training not supported for featurize model")

    def validation_step(self, batch, batch_idx):
        raise Exception("Validation not supported for featurize model")

    def test_step(self, batch, batch_idx):
        # Get outputs including features
        outputs = self.shared_step(batch, "test")

        # Validate outputs before storing
        if outputs is not None and len(outputs) == 3 and outputs[1] is not None:
            # Store outputs for epoch end
            self.test_step_outputs["outputs"].append(outputs)

        return outputs

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
        """Similar to training step"""
        try:
            x, y, instance_id = batch
        except:
            x, y, _, instance_id = batch

        # Get logits and features from model
        logit, features = self.model(x, get_features=True)

        # Create output tuple with all needed info
        output = (logit, features.detach(), instance_id)

        # Save features as individual files if needed
        if self.features_dir:
            for ids, f in zip(instance_id, features.cpu().detach().numpy()):
                try:
                    # Clean the image_id to remove .nii.gz and get base name
                    clean_id = ids.split('@')[0]  # First get impression_id
                    cleaned_image_id = self.df_metadata[self.df_metadata['impression_id'] == int(clean_id)]['image_id'].iloc[0]
                    cleaned_image_id = cleaned_image_id.replace('.nii.gz', '')  # Remove .nii.gz extension
                    feature_path = os.path.join(self.features_dir, f"{cleaned_image_id}.npy")
                    np.save(feature_path, f)
                except Exception as e:
                    print("[ERROR]", ids, str(e))
                    continue

        return output

    def shared_epoch_end(self, step_outputs, prefix):
        # Extract feature vectors and instance IDs
        all_features = []
        all_instance_ids = []

        # Collect outputs from all steps
        for outputs in step_outputs["outputs"]:
            if outputs is not None and len(outputs) > 0:
                # Extract features and instance ID
                _, feature_vector, instance_id = outputs
                if isinstance(instance_id, (list, tuple)):
                    all_instance_ids.extend(instance_id)
                    all_features.extend(feature_vector.cpu().numpy())
                else:
                    all_instance_ids.append(instance_id)
                    all_features.append(feature_vector.cpu().numpy())

        # Convert features to numpy array
        if len(all_features) > 0:
            all_features = np.vstack(all_features)

            # Create DataFrame with same number of rows for all columns
            df_data = {'instance_id': all_instance_ids}

            # Add feature columns
            for i in range(all_features.shape[1]):
                df_data[f'feature_{i}'] = all_features[:, i]

            # Create DataFrame
            df = pd.DataFrame(df_data)

            # Get impression ID from instance ID
            df["impression_id"] = df["instance_id"].apply(lambda x: x.split("@")[0])
            df["impression_id"] = df["impression_id"].astype(int)

            print(f"Processed {len(df)} instances with {all_features.shape[1]} features")

            # Create output path and save
            output_path = Path(self.features_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path / f"{prefix}_features.csv", index=False)
        else:
            print("Warning: No features collected during epoch")

        return {}

    def shared_epoch_end_rsna(self, step_outputs, split):
        df = pd.read_csv(self.cfg.dataset.csv_path)

        # match dicom datetime format
        # self.df['procedure_time'] = self.df['procedure_time'].apply(
        #     lambda x: x.replace('T', ' ')
        # )
        # df[SERIES_COL] = df[SERIES_COL]

        # get unique study id  by combining patient id and datetime
        # self.df["patient_datetime"] = self.df.apply(
        #     lambda x: f"{x.person_id}_{x.procedure_time}", axis=1
        # )
        # df["patient_datetime"] = df.apply(
        #     lambda x: f"{x.person_id}_{x.procedure_time}", axis=1
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
            #         self.cfg.dataset.dicom_dir, str(row["person_id"]) + ".tar"
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
