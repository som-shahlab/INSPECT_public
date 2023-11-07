from .data_module import DataModule
from .dataset_1d import Dataset1D, RSNADataset1D
from .dataset_2d import Dataset2D, RSNADataset2D

ALL_DATASETS = {
    "1d": Dataset1D,
    "2d": Dataset2D,
    "rsna": RSNADataset2D,
    "rsna_1d": RSNADataset1D,
}
