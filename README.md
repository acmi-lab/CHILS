# CHiLS

![CHiLS](fig19.jpeg)

This is the official implementation for [CHiLS: Zero-shot Image Classification with Hierarchical Label Sets](https://arxiv.org/abs/2302.02551). If you find this repository useful or use this code in your research, please cite the following paper: 

> Zachary Novack, Julian McAuley, Zachary Lipton, and Saurabh Garg. Chils: Zero-shot image classification with hierarchical label sets. In International Conference on Machine Learning (ICML), 2023.
```
@inproceedings{novack2023chils,
    title={CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets},
    author={Novack, Zachary and McAuley, Julian and Lipton, Zachary and Garg, Saurabh},
    year={2023},
    booktitle={International Conference on Machine Learning (ICML)}, 
}
```

There are three main steps for recreating the paper results:

1. Setting up the environment and datasets
2. Caching the CLIP-extracted features for each dataset and model
3. Running zero-shot inference


## Setting up the environment and datasets:
All requisite packages can be installed via the `environment.yml` file. For access to GPT-3 through OpenAI, you must have an account and save your access token in the environment variable `OPENAI_API_KEY`.

Besides ImageNet, CIFAR100 and Fashion-MNIST (which can be autoloaded through the `torchvision` API), each dataset can be downloaded through the standard websites for each: [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code), [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html), [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101), [Fruits360](https://www.kaggle.com/datasets/moltean/fruits), [Fashion1M](https://github.com/Cysu/noisy_label), [LSUN-Scene](https://www.yf.io/p/lsun), [ObjectNet](https://objectnet.dev/).
Dataset Notes:
- Both LSUN-Scene and Fashion1M must be configured into the `ImageFolder` format, wherein the directory has named folders for each class, each containing all the images. Due to compute constraints, for LSUN-Scene we use the validation data only and for Fashion1M we use the first two large image folders (i.e. `0` and `1`).

## Caching the CLIP-extracted features for each dataset and model:
Running `run.py` will use the variables specified in `config.yaml` and extract the features of a given dataset and CLIP model. In order to run this, the variable `data_loc` must be changed to the directory where your datasets are held.

## Running zero-shot inference:
Once the features are extracted, you may run `zshot.py` to generate the zero-shot inference results with CHiLS. For example, to generate the results with the GPT-generated label sets (which are provided for reproducibility) on Food-101, the command would be:

```
python zshot.py --dataset=food-101 --model=ClipViTL14 --experiment=gpt --label-set-size=10 --data-dir=[INSERT YOUR PATH HERE]
```

See the `src/constants.py` file for valid inputs for each argument in the command.



