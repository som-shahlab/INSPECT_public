import torch
import numpy as np
import pandas as pd
import tqdm
import os

from PIL import Image
from pathlib import Path
from ..constants import *
from .dataset_base import DatasetBase
from ..utils import read_tar_dicom


class Dataset2D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.transform = transform
        self.cfg = cfg

        self.df = pd.read_csv(cfg.dataset.csv_path)

        # match dicom datetime format
        self.df["procedure_time"] = self.df["procedure_time"].apply(
            lambda x: x.replace("T", " ")
        )

        # get unique patient_datetime id  by combining patient id and datetime
        self.df["patient_datetime"] = self.df.apply(
            lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
        )

        if self.split != "all":
            self.df = self.df[self.df["split"] == self.split]

        if self.split == "train":
            if cfg.dataset.sample_frac < 1.0:
                num_pdt = list(self.df["patient_datetime"].unique())
                num_sample = int(num_pdt * cfg.dataset.sample_frac)
                sampled_pdt = np.random.choice(num_pdt, num_sample, replace=False)
                self.df = self.df[self.df["patient_datetime"].isin(sampled_pdt)]

        # get all dicom files for a study
        self.all_instances = []
        for idx, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
            # # glob all paths
            # study_path = (
            #     Path(self.cfg.dataset.dicom_dir)
            #     / str(row["patient_id"])
            #     / str(row["procedure_time"])
            # )
            # slice_paths = study_path.glob("*.dcm")

            tar_content = read_tar_dicom(
                os.path.join(
                    self.cfg.dataset.dicom_dir, str(row["patient_id"]) + ".tar"
                )
            )
            prefix = "./" + str(row["procedure_time"]) + "/"
            slice_paths = [
                slice_path
                for slice_path in tar_content.keys()
                if slice_path.startswith(prefix)
            ]

            # each instance includes patient datetime, patient id , datetime, instance idx and path
            for slice_path in slice_paths:
                # instance_idx = last digits of path
                instance_idx = str(slice_path).split("/")[-1].replace(".dcm", "")
                self.all_instances.append(
                    [row["patient_datetime"], instance_idx, slice_path]
                )
        # print(self.all_instances)

    def __getitem__(self, index):
        # get slice row
        pdt, instance_idx, slice_path = self.all_instances[index]

        # read slice from dicom
        ct_slice = self.process_slice(slice_path=slice_path)

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1, 2, 0))
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
        else:
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert("RGB")
        x = self.transform(ct_slice)

        # check dimention
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h)
        x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor([0])

        return x, y, f"{pdt}@{instance_idx}"

    def __len__(self):
        return len(self.all_instances)


class RSNADataset2D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(cfg.dataset.csv_path)
        self.transform = transform
        self.cfg = cfg
        self.label_name = "pe_present_on_image"

        # only use positive CTs
        self.df = self.df[self.df.negative_exam_for_pe == 0]

        if self.split != "all":
            self.df = self.df[self.df["Split"] == self.split]

        if self.split == "train":
            if cfg.dataset.sample_frac < 1.0:
                study = list(self.df["patient_datetime"].unique())
                num_study = len(study)
                num_sample = int(num_study * cfg.dataset.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df["patient_datetime"].isin(sampled_study)]

        print(self.df[self.label_name].value_counts())
        self.labels = self.df[self.label_name].to_list()

    def __getitem__(self, index):
        # get slice row
        instance_info = self.df.iloc[index]

        # get slice info
        study_id = instance_info[STUDY_COL]
        series_id = instance_info[SERIES_COL]
        instance_id = instance_info[INSTANCE_COL]

        slice_path = (
            Path(self.cfg.dataset.dicom_dir)
            / study_id
            / series_id
            / f"{instance_id}.dcm"
        )
        ct_slice = self.process_slice(slice_path=slice_path)

        if ct_slice.sum() == 0:
            print(study_id)
            instance_id = "skip"

        if len(ct_slice.shape) == 4:
            print(study_id, series_id, instance_id)
            ct_slice = np.squeeze(ct_slice[0])

        # transform
        if ct_slice.shape[0] == 3:
            try:
                ct_slice = np.transpose(ct_slice, (1, 2, 0))
                ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
            except:
                print(ct_slice.shape)

        else:
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert("RGB")

        x = self.transform(ct_slice)

        # check dimention
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h)
        x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor([instance_info[self.label_name]]).float()

        return x, y, 0, instance_id

    def __len__(self):
        return len(self.df)

    def get_sampler(self):
        neg_class_count = (np.array(self.labels) == 0).sum()
        pos_class_count = (np.array(self.labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.labels]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler
