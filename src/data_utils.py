from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, Food101, EuroSAT, ImageNet
import torchvision
from torchvision import transforms
from typing import Callable, Optional, List
from torch.utils.data import Subset
import numpy as np
import torch
import torch.utils.data as data
from src.simple_utils import load_pickle
import pathlib
import json
import os
import logging 
import pickle
from PIL import Image
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from torchvision.datasets import ImageFolder
#from torchgeo.datasets import RESISC45

from wilds import get_dataset as get_dataset_wilds
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 dataset."""

import os
from typing import Callable, Dict, Optional, cast, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from torchgeo.datasets import NonGeoClassificationDataset
from torchgeo.datasets.utils import download_url, extract_archive


class RESISC45(NonGeoClassificationDataset):
    """RESISC45 dataset.

    The `RESISC45 <http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html>`__
    dataset is a dataset for remote sensing image scene classification.

    Dataset features:

    * 31,500 images with 0.2-30 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 45 scene classes, 700 images per class
    * images extracted from Google Earth from over 100 countries
    * images conditions with high variability (resolution, weather, illumination)

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. airplane
    1. airport
    2. baseball_diamond
    3. basketball_court
    4. beach
    5. bridge
    6. chaparral
    7. church
    8. circular_farmland
    9. cloud
    10. commercial_area
    11. dense_residential
    12. desert
    13. forest
    14. freeway
    15. golf_course
    16. ground_track_field
    17. harbor
    18. industrial_area
    19. intersection
    20. island
    21. lake
    22. meadow
    23. medium_residential
    24. mobile_home_park
    25. mountain
    26. overpass
    27. palace
    28. parking_lot
    29. railway
    30. railway_station
    31. rectangular_farmland
    32. river
    33. roundabout
    34. runway
    35. sea_ice
    36. ship
    37. snowberg
    38. sparse_residential
    39. stadium
    40. storage_tank
    41. tennis_court
    42. terrace
    43. thermal_power_station
    44. wetland

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/jproc.2017.2675998

    """

    url = "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
    md5 = "d824acb73957502b00efd559fc6cfbbb"
    filename = "NWPU-RESISC45.rar"
    directory = "NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }
    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new RESISC45 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits
        self.root = root
        self.download = download
        self.checksum = checksum
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"resisc45-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f"resisc45-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        label = cast(int, sample["label"].item())
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, target

# log = logging.getLogger(__name__)
log = logging.getLogger("app")

osj = os.path.join

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

class ImageNetDS(data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['Imagenet32_val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def get_breeds_mapping(dataset_name, data_dir): 
    path = f"{data_dir}/imagenet/imagenet_hierarchy/"

    if dataset_name.startswith("living17"): 
        ret = make_living17(path, split="good")
    elif dataset_name.startswith("entity13"):
        ret = make_entity13(path, split="good")
    elif dataset_name.startswith("entity30"):
        ret = make_entity30(path, split="good")
    elif dataset_name.startswith("nonliving26"):
        ret = make_nonliving26(path, split="good")

    label_mapping = get_label_mapping('custom_imagenet', np.concatenate((ret[1][0], ret[1][1]), axis=1)) 

    return label_mapping

def get_dataset(data_dir, dataset, train, transform): 

    Imagenet_Folder_with_indices = dataset_with_indices(ImageFolder)
    ImageFolder_with_indices = dataset_with_indices(folder.ImageFolder)
    ImageNetDS_with_indices = dataset_with_indices(ImageNetDS)
    c100_idx = dataset_with_indices(CIFAR100)
    fm_idx = dataset_with_indices(FashionMNIST)
    f101_idx = dataset_with_indices(Food101)
    r45_idx = dataset_with_indices(RESISC45)
    esat_idx = dataset_with_indices(EuroSAT)
    inet_idx = dataset_with_indices(ImageNet)

    if dataset.lower() == "cifar100":
        data = c100_idx(root = data_dir + "/cifar100/", train=False, transform=transform, download=True)    
    elif dataset.lower() == "imagenet-sketch":
        data = inet_idx(data_dir + "/imagenet/imagenet-sketch/sketch", transform = transform, split='val')
    elif dataset.lower() == "fruits360":
        data = Imagenet_Folder_with_indices(data_dir + "/fruits-360/Test", transform = transform)
    elif dataset.lower() == "food-101":
        data = f101_idx(data_dir, transform = transform, split='test') 
    elif dataset.lower() == "resisc45":
        data = r45_idx(data_dir + '/RESISC45', transforms = transform, split='val') 
    elif dataset.lower() == "eurosat":
        data = esat_idx(data_dir, transform = transform, download=True) 
    elif dataset.lower() == "lsun-scene":
        data = Imagenet_Folder_with_indices(data_dir + "/lsun/scene", transform = transform) 
    elif dataset.lower() in ["fashion1M", 'fashion1m']:
        data = Imagenet_Folder_with_indices(data_dir + "/fashion1M/clean_data", transform = transform)
    elif dataset.lower() == "imagenet":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenetv1/val", transform = transform)
    elif dataset.lower() == "objectnet":
        data = Imagenet_Folder_with_indices(data_dir + "/objectnet-1.0/images", transform = transform)
    elif dataset.lower() == "imagenet-c1":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/fog/1", transform = transform)
    elif dataset.lower() == "imagenet-c2":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/contrast/2", transform = transform)
    elif dataset.lower() == "imagenet-c3":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/snow/3", transform = transform)
    elif dataset.lower() == "imagenet-c4":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/gaussian_blur/4", transform = transform)
    elif dataset.lower() == "imagenet-c5":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenet-c/saturate/5", transform = transform)
    elif dataset.lower() == "imagenetv2":
        data = Imagenet_Folder_with_indices(data_dir + "/imagenet/imagenetv2/imagenetv2-matched-frequency-format-val", transform = transform)
    elif dataset.lower() == "office31-amazon":
        data = Imagenet_Folder_with_indices(data_dir + "/office31/amazon/images/", transform = transform)
    elif dataset.lower() == "office31-dslr":
        data = Imagenet_Folder_with_indices(data_dir + "/office31/dslr/images/", transform = transform)
    elif dataset.lower() == "office31-webcam":
        data = Imagenet_Folder_with_indices(data_dir + "/office31/webcam/images/", transform = transform)
    elif dataset.lower() == "officehome-product":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/Product/", transform = transform)
    elif dataset.lower() == "officehome-realworld":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/RealWorld/", transform = transform)
    elif dataset.lower() == "officehome-art":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/Art/", transform = transform)
    elif dataset.lower() == "officehome-clipart":
        data = Imagenet_Folder_with_indices(data_dir + "/officehome/Clipart/", transform = transform)  
    elif dataset.lower() == "fashion-mnist": 
        data = fm_idx(root = data_dir, train=False, transform=transform, download=True)
    else: 
        raise NotImplementedError("Please add support for %s dataset" % dataset)
    return data


def split_idx(y_true, num_classes = 1000): 

    classes_idx = []

    y_true = np.array(y_true)
    
    for i in range(num_classes): 
        classes_idx.append(np.where(y_true==i)[0])

    return classes_idx
