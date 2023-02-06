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
import json
from models.clip_models import *
from src.zshot_utils import *
from src.constants import *
from src.get_inputs import *

openai.api_key = os.environ.get("OPENAI_API_KEY")

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='nonliving26')
parser.add_argument("--domain", type=str, default='all')
parser.add_argument("--model", type=str, default='ClipViTL14')
parser.add_argument("--experiment", type=str, default='true')
parser.add_argument("--data-dir", type=str)
parser.add_argument("--mod-out-dir", type=str, default='')
parser.add_argument("--out-dir", type=str)
parser.add_argument("--label-set-size", type=int, default=10)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--rerun-gpt", action='store_true')
parser.add_argument("--use-gpt", action='store_true')
parser.add_argument("--best-poss", action='store_true')
parser.add_argument("--reweighter",type=str, default='normal')
parser.add_argument("--opt-templates", action='store_true')
parser.add_argument("--superclass-set-ens", action='store_true')



def check_args(args):
    if args.dataset not in DATASETS:
        raise NotImplementedError(f"Invalid dataset: {args.dataset}")
    if args.domain != "all" and args.dataset in DOMAINS.keys() and args.domain not in DOMAINS[args.dataset]:
        raise NotImplementedError(f"Invalid domain: {args.domain}")
    if args.model not in MODELS:
        raise NotImplementedError(f"Invalid model: {args.model}")
    if args.experiment not in EXPERIMENTS:
        raise NotImplementedError(f"Invalid experiment : {args.experiment}")
    if args.dataset not in TRUESETS and args.experiment == 'true':
        raise NotImplementedError(f"No ground-truth subsets for {args.dataset}")


if __name__ == '__main__':
    args = parser.parse_args()
    check_args(args)
    print(args)
    mod = eval(args.model)()
    if args.opt_templates:
        run = runBestTemplate
    if args.domain == 'all' and args.dataset in DOMAINS.keys():
        domains = DOMAINS[args.dataset]
        out = {}
        for domain in domains:
            args.domain = domain
            features, labels, sub2super, super_classes, indices = get_inputs(args)
            out_dom, _ = run(mod, features, labels, sub2super, super_classes, indices, args)
            args.rerun_gpt = False
            if args.opt_templates:
                for temp, d in out_dom.items():
                    if temp not in out:
                        out[temp] = {}
                    for k, v in d.items():
                        if k in out[temp].keys():
                            out[temp][k] += d[k] / len(domains)
                        else:
                            out[temp][k] = d[k] / len(domains)
            else:
                for k, v in out_dom.items():
                    if k in out.keys():
                        out[k] += out_dom[k] / len(domains)
                    else:
                        out[k] = out_dom[k] / len(domains)
        print(out)
    else:
        features, labels, sub2super, super_classes, indices = get_inputs(args)
        out, preds_d = run(mod, features, labels, sub2super, super_classes, indices, args)
        print(out)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    pref = f"/{args.dataset}-{args.model}-{args.experiment}-{args.label_set_size}-{args.reweighter}"
    if args.superclass_set_ens:
        pref = f"/{args.dataset}-{args.model}-{args.experiment}-{args.label_set_size}-{args.reweighter}-sse"
    with open(args.out_dir + pref + f".json", 'w') as f:
        json.dump(out, f, indent=2)
    if args.domain != 'all':
        with open(args.out_dir + pref + f"-outputs.pkl", 'wb') as f:
            pk.dump(preds_d, f)



