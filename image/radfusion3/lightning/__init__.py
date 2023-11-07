from .featurize_lightning_model import FeaturizeLightningModel
from .classification_lightning_model import ClassificationLightningModel

ALL_LIGHTNING_MODELS = {
    'extract': FeaturizeLightningModel,
    'classify': ClassificationLightningModel 
}
