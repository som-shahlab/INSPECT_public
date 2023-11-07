import torch
import numpy as np
import pandas as pd
import cv2

from ..constants import *
from .dataset_base import DatasetBase
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path


class Dataset2D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(cfg.dataset.csv_path)
        self.df['patient_datetime'] = self.df.apply(
            lambda x: f"{x.patient_id}_{x.procedure_time}",
            axis=1

        )
        self.transform = transform
        self.cfg = cfg

        if self.split != "all":
            self.df = self.df[self.df['split'] == self.split]

        if self.split == "train":
            if cfg.dataset.sample_frac < 1.0:
                pdt = list(self.df['patient_datetime'].unique())
                num_pdt = len(pdt)
                num_sample = int(num_pdt * cfg.dataset.sample_frac)
                sampled_pdt = np.random.choice(pdt, num_sample, replace=False)
                self.df = self.df[self.df['patient_datetime'].isin(sampled_pdt)]

        self.all_instances = []
        for idx, row in self.df.iterrows():
            for instance_idx in range(row.num_slices):
                self.all_instances.append(
                    [row['patient_datetime'], instance_idx]
                )

    def __getitem__(self, index):

        # get slice row
        pdt, instance_idx = self.all_instances[index]

        # get slice info 
        ct_slice = self.process_numpy(
            Path(self.cfg.dataset.numpy_dir) / f"{pdt}.npy", instance_idx
        )

        # transform
        if ct_slice.shape[0] == 3:
            try:
                ct_slice = np.transpose(ct_slice, (1,2,0))
                ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
            except:
                print(ct_slice.shape)

        else: 
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert('RGB')
        
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
        self.label_name = 'pe_present_on_image'

        # only use positive CTs
        self.df = self.df[self.df.negative_exam_for_pe == 0]

        if self.split != "all":
            self.df = self.df[self.df['Split'] == self.split]

        if self.split == "train":
            if cfg.dataset.sample_frac < 1.0:
                study = list(self.df['patient_datetime'].unique())
                num_study = len(study)
                num_sample = int(num_study * cfg.dataset.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df['patient_datetime'].isin(sampled_study)]

        print(self.df[self.label_name].value_counts())
        self.labels = self.df[self.label_name].to_list()

    def __getitem__(self, index):

        # get slice row
        instance_info = self.df.iloc[index]

        # get slice info 
        study_id = instance_info[STUDY_COL]
        series_id = instance_info['patient_datetime']
        instance_id = instance_info[INSTANCE_COL]

        slice_path= Path(self.cfg.dataset.dicom_dir) / study_id / series_id / f"{instance_id}.dcm"
        ct_slice = self.process_slice(
            slice_path = slice_path
        )

        if ct_slice.sum() == 0:
            print(study_id)
            instance_id = 'skip'

        if len(ct_slice.shape) == 4:
            print(study_id, series_id, instance_id)
            ct_slice = np.squeeze(ct_slice[0])

        # transform
        if ct_slice.shape[0] == 3:
            try:
                ct_slice = np.transpose(ct_slice, (1,2,0))
                ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
            except:
                print(ct_slice.shape)

        else: 
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert('RGB')
        
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
        return len(df)

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
