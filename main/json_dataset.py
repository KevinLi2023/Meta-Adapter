#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter
from typing import List, Union
import json
from torchvision import transforms

class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, data_name, data_dir, split, transform):
        self.data_name = data_name
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self._construct_imdb()
    
    def read_json(self, filename: str) -> Union[list, dict]:
      """read json files"""
      with open(filename, "rb") as fin:
          data = json.load(fin, encoding="utf-8")
      return data

    
    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self.split))
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)
        return self.read_json(anno_path)

    def get_imagedir(self):
        if self.data_name == "cub":
            return os.path.join(self.data_dir, "images")
        elif self.data_name == "nabirds":
            return os.path.join(self.data_dir, "images")
        elif self.data_name == "stanford_cars":
            return os.path.join(self.data_dir, "Images")
        elif self.data_name == "stanford_dogs":
            return os.path.join(self.data_dir, "Images")
        elif self.data_name == "fgvc_flowers":
            return self.data_dir
        else:
            raise NotImplementedError()

    def _construct_imdb(self):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})
        
        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return 196

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self.split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        img = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        target = self._imdb[index]["class"]
        if self.transform != None:
            img = self.transform(img)
        if self.split == "train":
            index = index
        else:
            index = f"{self.split}{index}"
    
        return img, target

    def __len__(self):
        return len(self._imdb)

def get_fgvc_data(name, evaluate=True, batch_size=64):
    root = '/path/to/FGVC/' + name
    transform1 = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            JSONDataset(data_name=name, data_dir=root, split="train",transform=transform1),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            JSONDataset(data_name=name, data_dir=root, split="test",
                transform=transform2),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            JSONDataset(data_name=name, data_dir=root, split="train",
                transform=transform1),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            JSONDataset(data_name=name, data_dir=root, split="val",
                transform=transform2),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)
    return train_loader, val_loader

