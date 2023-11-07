import torch
import pydicom
import numpy as np
import pandas as pd
import cv2
import h5py

from ..constants import *
from torch.utils.data import Dataset
from pathlib import Path


class DatasetBase(Dataset):
    def __init__(self, cfg, split="train", transform=None):

        self.cfg = cfg
        self.transform = transform
        self.split = split
        self.hdf5_dataset = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_hdf5(self, key, hdf5_path, slice_idx=None):
        if self.hdf5_dataset is None: 
            self.hdf5_dataset = h5py.File(hdf5_path, 'r')
       
        if slice_idx is None: 
            arr = self.hdf5_dataset[key][:]
        else: 
            arr = self.hdf5_dataset[key][slice_idx]
        return arr

    def read_dicom(self, file_path: str, resize_size=None, channels=None):
        if resize_size is None:
            resize_size = self.cfg.dataset.transform.resize_size
        if channels is None:
            channels = self.cfg.dataset.transform.channels

        # read dicom
        dcm = pydicom.dcmread(file_path)
        try: 
            pixel_array = dcm.pixel_array
        except: 
            print(file_path)
            if channels == 'repeat':
                pixel_array = np.zeros(
                    (resize_size,
                    resize_size)
                )
            else:
                pixel_array = np.zeros(
                    (3,
                    resize_size,
                    resize_size)
                )

        # rescale
        try:
            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope
        except:
            intercept = 0 
            slope =1 

        pixel_array = pixel_array * slope + intercept

        # resize
        if resize_size != pixel_array.shape[-1]:
            pixel_array = cv2.resize(
                pixel_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )

        return pixel_array

    def windowing(self, pixel_array: np.array, window_center: int, window_width: int):

        lower = window_center - window_width // 2
        upper = window_center + window_width // 2
        pixel_array = np.clip(pixel_array.copy(), lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower)

        return pixel_array

    def process_numpy(self, numpy_path, idx):

        slice_array = np.load(numpy_path)[idx]

        resize_size = self.cfg.dataset.transform.resize_size
        channels = self.cfg.dataset.transform.channels

        if resize_size != slice_array.shape[-1]:
            slice_array = cv2.resize(
                slice_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )

        # window
        if self.cfg.dataset.transform.channels == 'repeat':
            
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400), # MEDIASTINAL window
            ]  
            ct_slice = np.stack(ct_slice)

        return ct_slice
    

    def process_slice(self, slice_info: pd.Series = None, dicom_dir: Path = None, slice_path: str = None):
        """process slice with windowing, resize and tranforms"""

        if slice_path is None:
            slice_path =  dicom_dir / slice_info[INSTANCE_PATH_COL]
        slice_array = self.read_dicom(slice_path)

        # window
        if self.cfg.dataset.transform.channels == 'repeat':
            
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400), # MEDIASTINAL window
            ]  
            ct_slice = np.stack(ct_slice)

        return ct_slice

    def fix_slice_number(self, df: pd.DataFrame):

        num_slices = min(self.cfg.dataset.num_slices, df.shape[0])
        if self.cfg.dataset.sample_strategy == "random":
            slice_idx = np.random.choice(
                np.arange(df.shape[0]), replace=False, size=num_slices
            )
            slice_idx = list(np.sort(slice_idx))
            df = df.iloc[slice_idx, :]
        elif self.cfg.dataset.sample_strategy == "fix":
            df = df.iloc[:num_slices, :]
        else:
            raise Exception("Sampling strategy either 'random' or 'fix'")
        return df

    def fix_series_slice_number(self, series):

        num_slices = min(self.cfg.dataset.num_slices, series.shape[0])
        if num_slices == self.cfg.dataset.num_slices:
            if self.cfg.dataset.sample_strategy == "random":
                slice_idx = np.random.choice(
                    np.arange(series.shape[0]), replace=False, size=num_slices
                )
                slice_idx = list(np.sort(slice_idx))
                features = series[slice_idx, :]
            elif self.cfg.dataset.sample_strategy == "fix":
                pad = int((series.shape[0] - num_slices) / 2)    # select middle slices
                start = pad
                end = pad+num_slices
                features = series[start:end, :]
            else:
                raise Exception("Sampling strategy either 'random' or 'fix'")
            mask = np.ones(num_slices)
        else:
            mask = np.zeros(self.cfg.dataset.num_slices)
            mask[:num_slices] = 1
            shape = [self.cfg.dataset.num_slices] + list(series.shape[1:])
            features = np.zeros(shape)

            features[:num_slices] = series

        return features, mask

    def fill_series_to_num_slicess(self, series, num_slices):
        x = torch.zeros(()).new_full((num_slices, *series.shape[1:]), 0.0)
        x[: series.shape[0]] = series
        return x
