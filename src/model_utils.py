import torch.nn as nn
import torchvision
import torch 

import os
from os.path import join
from torchvision.models.feature_extraction import create_feature_extractor

# from models import * 
import logging 

from models.clip_models import *

log = logging.getLogger("app")


def get_model(arch, dataset, num_classes=1000, pretrained=True, retrain = False, extract_features = False, work_dir = None): 

    if arch =="ClipViTL14": 
        net = ClipViTL14(num_classes)
    elif arch =="ClipViTB32": 
        net = ClipViTB32(num_classes)
    elif arch =="ClipViTB16": 
        net = ClipViTB16(num_classes)
    elif arch =="ClipRN50x4": 
        net = ClipRN50x4(num_classes)
    elif arch =="ClipRN101": 
        net = ClipRN101(num_classes)
    elif arch =="ClipRN50": 
        net = ClipRN50(num_classes)

    else: 
        raise NotImplementedError("Net %s is not implemented" % arch)

    if arch in ('ClipViTL14', 'ClipViTB32', 'ClipViTB16', 'ClipRN50x4', 'ClipRN101', 'ClipRN50'): 
        assert retrain == False

        if extract_features: 
            log.info(f"Only extracting the features of a pre-trained {arch} model")
            for param in net.parameters():
                param.requires_grad = False

    return net