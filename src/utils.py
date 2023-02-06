import yaml
import hashlib
import time
import importlib
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


def get_dict_hash(dictionary: dict) -> str:
    dhash = hashlib.md5()
    dump = yaml.dump(dictionary)
    encoded = dump.encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_class_name(module_class_string, split=None):
    module_name, class_name = module_class_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(
        class_name, module_name
    )
    cls = getattr(module, class_name)
    name = cls.name
    if split is not None:
        name += "_" + split
    return name


def load_config(config_file: str) -> dict:
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    return config


def get_random_seed() -> int:
    return int(time.time() * 256) % (2 ** 32)


def filter_config(config: DictConfig) -> dict:
    def is_special_key(key: str) -> bool:
        return key[0] == "_" and key[-1] == "_"

    primitive_config = OmegaConf.to_container(config)

    filt = {
        k: v
        for k, v in primitive_config.items()
        if (not OmegaConf.is_interpolation(config, k))
        and (not is_special_key(k))
        and v is not None
    }
    return filt


def log_hyperparams(config: DictConfig, trainer: Trainer) -> None:
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    for key in ["trainer", "models", "datamodule"]: 
        hparams[key] = filter_config(config[key])
    
    for key in ["source_dataset", "arch", "num_classes"]: 
        hparams[key] = config[key]

    trainer.logger.log_hyperparams(hparams)


def add_to_odict(odict, item):
    if item not in odict:
        ind = len(odict)
        odict[item] = ind
