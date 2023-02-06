import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot

from typing import List, Optional
from src.model_utils import *
import logging 
import wandb
import numpy as np

log = logging.getLogger("app")

class EvalNet(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet50",
        retrain: bool = False,
        base_task: str = "Imagenet",
        pretrained: bool = True,
        target_dataset: List[str] = [], 
        work_dir: str = ".",
        max_epochs: int = 1,
        hash: Optional[str] = None
    ):
        super().__init__()

        self.arch = arch
        self.model = get_model(arch = arch, dataset = base_task , pretrained= pretrained, retrain=False, extract_features=True, work_dir=work_dir)
        
        
        self.model.init_text(base_task.lower())
        self.target_dataset = target_dataset

        # self.pred_acc = nn.ModuleList([Accuracy() for _ in self.target_dataset])

        # self.confusion_matrix = ConfusionMatrix(1000)
        
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, _ = batch[:2]
        return self.model(x)

    def process_batch(self, batch, stage="train", dataloader_idx=0):
        x, y, idx = batch[:3]

        output = self.forward(x)
        logits = output["logits"]
        features = torch.flatten(output["features"],1)
        
        _, pred_idx = torch.max(logits, dim=1)

        return  y, pred_idx, idx, features


    def training_step(self, batch, batch_idx: int):
        _ = self.process_batch(batch, "pred")
        
        return None

        # pass

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        labels, outputs, idx, features = self.process_batch(batch, "pred",dataloader_idx)
        
        return labels, outputs, idx, features

    def test_epoch_end(self, outputs_list):
        
        
        labels = torch.cat([x[0] for x in outputs_list])
        outputs = torch.cat([x[1] for x in outputs_list])
        idx = torch.cat([x[2] for x in outputs_list])
        features = torch.cat([x[3] for x in outputs_list])

        if not os.path.exists(self.work_dir + f"/{self.arch}"):
            os.mkdir(self.work_dir + f"/{self.arch}")
        np.savez(self.work_dir + f"/{self.arch}/conf_" + self.target_dataset.lower() +".npz",\
            labels = labels.detach().cpu().numpy(),\
            outputs = outputs.detach().cpu().numpy(),\
            indices = idx.detach().cpu().numpy(),\
            features = features.detach().cpu().numpy())

    def configure_optimizers(self):
        pass