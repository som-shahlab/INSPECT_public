import torch

from omegaconf import OmegaConf
from . import models
from . import lightning
from . import data
from . import transforms


def build_data_module(cfg):
    return data.data_module.DataModule(cfg)

def build_dataset(cfg):
    if cfg.dataset.type in data.ALL_DATASETS:
        return data.ALL_DATASETS[cfg.dataset.type]
    else:
        raise NotImplementedError(
            f"Dataset not implemented for {cfg.dataset.type} \n" + 
            f"Available datasets: {datasets.ALL_DATASETS.keys()}"
        )

def build_lightning_model(cfg, ckpt=None):
    if cfg.stage in lightning.ALL_LIGHTNING_MODELS:
        model = lightning.ALL_LIGHTNING_MODELS[cfg.stage]
    else:
        raise NotImplementedError(
            f"Lightning model not implemented for {cfg.lightning_model} \n" +
            f"Available lightning models: {lightning.ALL_LIGHTNING_MODELS.keys()}"
        )
    if ckpt is not None:
        return model.load_from_checkpoint(ckpt)
    return model(cfg)


def build_model(cfg):
    if cfg.model.type in models.ALL_MODELS:
        if cfg.stage == 'extract':
            num_class = 1 # dummy class number
        else:
            num_class = 1 # TODO 
        return models.ALL_MODELS[cfg.model.type](cfg, num_class)
    else:
        raise NotImplementedError(
            f"Model not implemented for {cfg.model.type} \n" +
            f"Available models: {models.ALL_MODELS.keys()}"
        )

def build_transformation(cfg, split):
    if cfg.model.type == 'model_1d':
        return None

    if cfg.model.pretrain_type in transforms.ALL_TRANSFORMS: 
        model_transform = transforms.ALL_TRANSFORMS[cfg.model.pretrain_type]
    else:
        raise NotImplementedError(
            f"Transformation not implemented for {cfg.model.pretrain_type} \n" +
            f"Available tranforms: {transforms.ALL_TRANSFORMS}"
        )
    
    if cfg.stage in model_transform.keys():
        stage_transform = model_transform[cfg.stage]
    else:
        raise NotImplementedError(
            f"Transformation not implemented for stage: {cfg.stage} \n" +
            f"Tranformations availible for stages: {model_transform.keys()}"
        )

    if split in stage_transform.keys():
        split_transform = stage_transform[split]
    else:
        raise NotImplementedError(
            f"Transformation not implemented for split: {split} \n" +
            f"Tranformations availible for split: {stage_transform.keys()}"
        )

    if 'transform' in cfg.model: 
        return split_transform(**cfg.model.transform)
    elif 'transform' in cfg.dataset: 
        return split_transform(**cfg.dataset.transform)
    else:
        return split_transform()



def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]

    if 'optimizer' in cfg:
        optimizer_name = cfg.optimizer.name
        del cfg.optimizer.name
        optimizer_fn = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_fn(params, lr=cfg.lr, **cfg.optimizer)
        #cfg.train.optimizer.name = optimizer_name
        return optimizer
    else: 
        return None


def build_loss(cfg):
    # get loss function
    if 'loss' in cfg:
        loss_name = cfg.loss.loss_fn
        del cfg.loss.loss_fn
        loss_fn = getattr(torch.nn, loss_name)
        loss_function = loss_fn(**cfg.loss)
        return loss_function
    else: 
        return None
