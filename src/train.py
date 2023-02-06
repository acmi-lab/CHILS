import hydra
from pytorch_lightning.utilities import seed
import wandb
import logging
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from src.utils import log_hyperparams
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


# log = logging.getLogger(__name__)
log = logging.getLogger("app")
log.setLevel(logging.INFO)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def train(config: DictConfig):
    log.info(f"Fixing the seed to <{config.seed}>")
    seed_everything(int(config.seed))

    log.info(f"Instantiating logger <{config.logger._target_}>")
    # logger: WandbLogger = hydra.utils.instantiate(config.logger)

    print(config.work_dir)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath =f"{config.work_dir}/model_checkpoint_kshot/",\
    #     filename = f"{config.target_dataset}_{config.k}",\
    #     save_last=True)


    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        # logger=logger,
        num_sanity_val_steps=0,
        # callbacks=[EarlyStopping(monitor="pred_acc", mode='max', patience=10)]
    )

    log.info(f"Instantiating model <{config.models._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.models)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    # log.info("Starting training!")
    # trainer.fit(model, datamodule)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    wandb.finish()
