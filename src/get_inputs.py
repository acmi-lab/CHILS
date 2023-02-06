from .zshot_utils import *
import argparse
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
import time
from torchvision.io import read_image, ImageReadMode
import torch
import numpy as np
import torchvision
import nltk
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
import torch.nn as nn
import pickle as pk
import matplotlib.pyplot as plt
import time
import openai
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from .constants import *
import json
from robustness.tools.breeds_helpers import BreedsDatasetGenerator, print_dataset_info, ClassHierarchy


def get_inputs(args):
    feature_path = f"{args.model}"
    if args.mod_out_dir != '':
        feature_path = args.mod_out_dir + "/" + feature_path
    features_file= feature_path + "/conf_%s.npz"
    if args.dataset in ['nonliving26', 'living17', 'entity13', 'entity30', 'inet1', 'inet2', 'inet3', 'inet4', 'inet5', 'inet6']:
        hier = ClassHierarchy(f"{args.data_dir}/imagenet/imageNet_hierarchy/")
        DG = BreedsDatasetGenerator(f"{args.data_dir}/imagenet/imageNet_hierarchy/")
        data = np.load(features_file % args.domain)
        idir = get_idir(args.data_dir, args.domain)
        features, labels, outputs, indices = prep_data(data, args, idir)
        if 'inet' in args.dataset:
            ret = DG.get_superclasses(level=int(args.dataset[-1]),
                Nsubclasses=None,
                #Nsubclasses=2,
                ancestor=None,
                balanced=False
            )
        else:
            ret = eval(f"make_{args.dataset}")(f"{args.data_dir}/imagenet/imageNet_hierarchy/", split='good')
        info_df = print_dataset_info(ret[0],ret[1],ret[2], hier.LEAF_NUM_TO_NAME)
        if 'inet' not in args.dataset:
            info_df['subclasses'] = info_df['subclasses (source)'] + info_df['subclasses (target)']
        info_df['dumm'] = info_df['superclass'].apply(lambda x: 'dummy' in x)
        info_df.loc[info_df['dumm'], 'superclass'] = info_df.loc[info_df['dumm'], 'subclasses'].apply(lambda x: re.sub(r'\([^)]*\)', '', x[0]))
        super_classes = [x.split(",")[0].strip() for x in info_df['superclass']]
        sup2sub = {
            k: v for k, v in zip(
                super_classes, info_df['subclasses'].apply(
                    lambda z: [
                        re.sub(r'\([^)]*\)', '', y).strip() for y in sorted(
                            z, key=lambda x: int(re.search(r'\([^)]*\)', x).group(0).strip("()"))
                        )
                    ]
                )
            )
        }

    elif args.dataset == 'office31':
        data = np.load(features_file % f'office31-{args.domain}')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/office31/{args.domain}/images/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/office31/{args.domain}/images/')
        breeds_idx = {" ".join(k.split("_")): v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())

    elif args.dataset == 'officehome':
        data = np.load(features_file % f'officehome-{args.domain}')
        dom_map = {'art': 'Art', 'clipart': 'Clipart', 'product': 'Product', 'realworld': 'RealWorld'}
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/officehome/{dom_map[args.domain]}/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/officehome/{dom_map[args.domain]}/')
        breeds_idx = {" ".join(k.split("_")): v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())

    elif args.dataset == 'fashion1M':
        data = np.load(features_file % 'fashion1M'.lower())
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/fashion1M/clean_data/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/fashion1M/clean_data/')
        breeds_idx = {" ".join(k.split("_")): v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())

    elif args.dataset == 'fashion-mnist':
        data = np.load(features_file % 'fashion-mnist')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        fm_idx = dataset_with_indices(datasets.FashionMNIST)
        datad = fm_idx(root = args.data_dir, train=False, download=True)
        breeds_idx = datad.class_to_idx
        breeds_idx['T-shirt'] = breeds_idx.pop('T-shirt/top')
        breeds_idx = {k: v for k,v in sorted(breeds_idx.items(), key=lambda x: x[1])}
        super_classes = list(breeds_idx.keys())

    elif args.dataset == 'lsun-scene':
        data = np.load(features_file % 'lsun-scene')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/lsun/scene/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/lsun/scene/')
        breeds_idx = {" ".join(k.split("_")): v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())
    
    elif args.dataset == 'resisc45':
        data = np.load(features_file % 'resisc45')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/RESISC45/NWPU-RESISC45/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/RESISC45/NWPU-RESISC45/')
        breeds_idx = {" ".join(k.split("_")): v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())
    
    elif args.dataset == 'eurosat':
        data = np.load(features_file % 'eurosat')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/eurosat/2750/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/eurosat/2750/')
        breeds_idx = {EUROFILE2NAME[k]: v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())

    elif args.dataset == 'food-101':
        data = np.load(features_file % 'food-101')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/food-101/images/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/food-101/images/')
        breeds_idx = {" ".join(k.split("_")): v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())

    elif args.dataset == 'cifar20':
        data = np.load(features_file % 'cifar100')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        super_classes = CIFAR20_COARSE
        sup2sub = {}
        target_map = {i: x for i, x in enumerate(CIFAR20_LABELS)}
        for i, cl in enumerate(CIFAR20_LABELS):
            if sup2sub.get(super_classes[cl]) is None:
                sup2sub[super_classes[cl]] = [CIFAR20_FINE[i]]
            else:
                sup2sub[super_classes[cl]].append(CIFAR20_FINE[i])
        labels = np.array([target_map[x] for x in labels])

    elif args.dataset == 'fruits360':
        data = np.load(features_file % 'fruits360')
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/fruits-360/Training/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/fruits-360/Training/')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        super_classes = list(np.unique(list(FRUITS360_OG2SUP.values())))
        sup_idx = {x: i for i,x in enumerate(super_classes)}
        orig2sup_idx = {v: sup_idx[FRUITS360_OG2SUP[k]] for k,v in breeds_idx.items()}
        sup2sub = {}
        for k,v in FRUITS360_OG2SUP.items():
            if sup2sub.get(FRUITS360_OG2SUP[k]) is None:
                sup2sub[FRUITS360_OG2SUP[k]] = [FRUITS360_OG2SUB[k]]
            else:
                if FRUITS360_OG2SUP[k] not in sup2sub[FRUITS360_OG2SUP[k]]:
                    sup2sub[FRUITS360_OG2SUP[k]].append(FRUITS360_OG2SUB[k])
        labels = np.array([orig2sup_idx[x] for x in labels])

    elif args.dataset == 'objectnet':
        with open(f"{args.data_dir}/objectnet-1.0/mappings/folder_to_objectnet_label.json", 'r') as f:
            fold2obj = json.load(f)
        datad = Imagenet_Folder_with_indices(f'{args.data_dir}/objectnet-1.0/images/')
        _, breeds_idx = datad.find_classes(f'{args.data_dir}/objectnet-1.0/images/')
        breeds_idx = {fold2obj[k]: v for k,v in breeds_idx.items()}
        super_classes = list(breeds_idx.keys())
        sup_idx = {k.split(",")[0]:i for i,k in enumerate(OBJECTNET_MAP.keys())}
        super_classes = list(sup_idx.keys())
        coarse_map = {}
        for k, v in OBJECTNET_MAP.items():
            for sub in v:
                coarse_map[breeds_idx[sub]] = sup_idx[k.split(",")[0]]
        sup2sub = {}
        for k, v in OBJECTNET_MAP.items():
            sup2sub[k.split(",")[0]] = v
        data = np.load(features_file % 'objectnet')
        features, labels, outputs, indices = data["features"], data["labels"], data["outputs"], data["indices"]
        idx = []
        for k in coarse_map.keys():
            idx += list(np.where(labels == k)[0])

        features = features[idx]
        labels = labels[idx]
        labels = np.array([coarse_map[x] for x in labels])
    



    if args.experiment in ['true', 'true_lin', 'true_wsup']:
        assert args.dataset in TRUESETS
        if args.experiment == 'true_wsup':
            sup2sub = {k: (v + [k] if k not in v else v) for k,v in sup2sub.items()}
        return features, labels, sup2sub, super_classes, indices
    elif args.experiment in ['gpt', 'gpt_lin', 'gpt_wosup']:
        if args.rerun_gpt:
            syn_d = get_cleaned_gpt_sets(super_classes, args.label_set_size, temp=args.temp, context=CONTEXTS.get(args.dataset, None))
            with open(f"label_sets/{args.dataset}-{args.label_set_size}.json", "w") as f:
                json.dump(syn_d, f, indent=3)
        else:
            try:
                with open(f"label_sets/{args.dataset}-{args.label_set_size}.json", "r") as f:
                    syn_d = json.load(f)
            except:
                if args.use_gpt:
                    syn_d = get_cleaned_gpt_sets(super_classes, args.label_set_size, temp=args.temp, context=CONTEXTS.get(args.dataset, None))
                    with open(f"label_sets/{args.dataset}-{args.label_set_size}.json", "w") as f:
                        json.dump(syn_d, f, indent=3)
                else:
                    raise ValueError(f"No existing label set found for {args.dataset}-{args.label_set_size}")
        if args.experiment == 'gpt_wosup':
            for k, v in syn_d.items():
                if k in v:
                    v.remove(k)
                syn_d[k] = v
        return features, labels, syn_d, super_classes, indices
    else:
        assert args.experiment == 'true_noise'
        assert args.dataset in ['nonliving26', 'living17', 'entity13', 'entity30']
        syn_d = get_breeds_all_subs(hier, DG, args.dataset)
        return features, labels, syn_d, super_classes, indices



