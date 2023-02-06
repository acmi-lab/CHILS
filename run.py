import hydra
import os
from os.path import join
from omegaconf import DictConfig, OmegaConf
from src.train import train
from src.utils import filter_config, get_dict_hash
from src.simple_utils import load_pickle, dump_pickle

@hydra.main(config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    # extract data and model experiment info to group runs
    group_dict = dict(filter_config(config.datamodule), **filter_config(config.models))
    # group_dict["name"] = get_class_name(config.datamodule._target_, "train")

    group_hash = get_dict_hash(group_dict)
    config.logger.group = group_hash
    print(group_dict)

    if not os.path.isdir(config.log_dir):
        os.mkdir(config.log_dir)

    hash_dict_fname = join(config.log_dir, "hash_dict.pkl")

    if os.path.isfile(hash_dict_fname):
        hash_dict = load_pickle(hash_dict_fname)
    else:
        hash_dict = dict()

    hash_dict[group_hash] = group_dict
    dump_pickle(hash_dict, hash_dict_fname)

    raw_path = join(config.log_dir, "raw")
    if not os.path.isdir(raw_path):
        os.mkdir(raw_path)

    # start training
    train(config)


if __name__ == "__main__":
    main()
