import torch.nn as nn
from . import vision_backbones


class Model2D(nn.Module):
    def __init__(self, cfg, num_class=1, **kwargs):
        super(Model2D, self).__init__()

        # define cnn model
        model_function = getattr(vision_backbones, cfg.model.model_name)
        model_kwargs = {}
        if hasattr(cfg.model, 'checkpoint_path'):
            model_kwargs['checkpoint_path'] = cfg.model.checkpoint_path
        self.model, self.feature_dim = model_function(**model_kwargs)
        self.classifier = nn.Linear(self.feature_dim, num_class)
        self.cfg = cfg
        self.get_features = cfg.get_features

    def forward(self, x, mask=None, get_features=False):
        x = self.model(x)
        pred = self.classifier(x)
        if get_features:
            return pred, x
        else:
            return pred
