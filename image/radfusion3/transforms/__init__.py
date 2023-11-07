#from . import dinov1, dinov2
from . import dinov1, dinov2, supervised

ALL_TRANSFORMS = {
    "dinov1": {
        "pretrain":{
            'train': dinov1.DataAugmentationDINOv1,
            'val': dinov1.DataAugmentationDINOv1,
        },
        "extract": {
            'train': dinov2.make_classification_eval_transform,
            'val': dinov2.make_classification_eval_transform,
            'test': dinov2.make_classification_eval_transform,
            'all': dinov2.make_classification_eval_transform,
        },
    },
    "dinov2": {
        "classify": {
            'train': supervised.make_classification_train_transform,
            'val': supervised.make_classification_eval_transform,
            'test': supervised.make_classification_eval_transform,
            'all': supervised.make_classification_eval_transform,
        },
        "pretrain":{
            'train': dinov2.DataAugmentationDINOv2,
            'val': dinov2.DataAugmentationDINOv2,
        },
        "extract": {
            'train': dinov2.make_classification_eval_transform,
            'val': dinov2.make_classification_eval_transform,
            'test': dinov2.make_classification_eval_transform,
            'all': dinov2.make_classification_eval_transform,
        },
    },
    "supervised": {
        "classify": {
            'train': supervised.make_classification_train_transform,
            'val': supervised.make_classification_eval_transform,
            'test': supervised.make_classification_eval_transform,
            'all': supervised.make_classification_eval_transform,
        },
        "extract": {
            'train': supervised.make_classification_eval_transform,
            'val': supervised.make_classification_eval_transform,
            'test': supervised.make_classification_eval_transform,
            'all': supervised.make_classification_eval_transform,
        },
    },
    "resnext": {
        "extract": {
            'train': supervised.make_classification_eval_transform,
            'val': supervised.make_classification_eval_transform,
            'test': supervised.make_classification_eval_transform,
            'all': supervised.make_classification_eval_transform,
        },
    }
}
