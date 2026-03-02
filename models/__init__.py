def build_model(model_type='direct', **kwargs):
    from .dtiqa import DTIQA
    model = DTIQA(**kwargs)
    return model, None

from .dtiqa import DTIQA
from .backbone import build_backbone
from .components import *
