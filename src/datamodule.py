import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
import numpy as np
from src.data_utils import *
import logging 
import clip


log = logging.getLogger("app")

all_classifiers = {
    "ClipViTB32": None
}

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        target_dataset: str = "Imagenet",
        batch_size: int = 128,
        num_classes: int = 1000,
        clip_transform: str = 'ClipViTL14'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.target_dataset = [target_dataset]
        self.num_classes = num_classes
        self.test_transform = []

        if clip_transform == 'ClipViTL14':
            _, preprocess = clip.load("ViT-L/14@336px", device='cuda')
            self.test_transform.append(preprocess)
        elif clip_transform == 'ClipViTB32':
            _, preprocess = clip.load("ViT-B/32", device='cuda')
            self.test_transform.append(preprocess)
        elif clip_transform == 'ClipViTB16':
            _, preprocess = clip.load("ViT-B/16", device='cuda')
            self.test_transform.append(preprocess)
        elif clip_transform == 'ClipRN50x4':
            _, preprocess = clip.load("RN50x4", device='cuda')
            self.test_transform.append(preprocess)
        elif clip_transform == 'ClipRN101':
            _, preprocess = clip.load("RN101", device='cuda')
            self.test_transform.append(preprocess)
        elif clip_transform == 'ClipRN50':
            _, preprocess = clip.load("RN50", device='cuda')
            self.test_transform.append(preprocess)
        else:    
            raise NotImplementedError("invalid CLIP model")

    def setup(self, stage: Optional[str] = None):

        log.info("Creating validation data ... ")

        self.test_dataset = []

        for i, dataset in enumerate(self.target_dataset): 

            self.data = get_dataset(\
                data_dir = self.data_dir,\
                dataset = dataset, \
                train = False,\
                transform = self.test_transform[i])
 
            self.test_dataset.append(self.data)

        log.info("Done")

    def train_dataloader(self):
        
        dataloaders = DataLoader(
                self.train_dataset, 
                batch_size=128, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True
        )
        
        return dataloaders

    def test_dataloader(self):
        
        dataloaders = []
        
        for i, dataset in enumerate(self.target_dataset): 
            dataloaders.append(DataLoader(
                self.test_dataset[i], 
                batch_size=128, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            ))
        
        return dataloaders
