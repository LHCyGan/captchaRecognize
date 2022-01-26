# -*- encoding:utf-8 -*-
# author: liuheng
import os
import glob

import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from torchvision import transforms as T
from torch.utils.data import Dataset

from utils import str_to_tensor, parse_label_map_c2i


class ImageDataset(Dataset):
    def __init__(self, path: str='./data', transform: T=None,
                 extensions: List[str]=None, maxLength: int=None,
                 placeholder: bool=True):
        """
        图片数据集: label_md5.jpg
        :param path: 目录
        :param transform: 预处理
        :param extensions: 图片扩展名
        :param maxLength: 标签填充到最大长度，为None不填充
        :param placeholder: 占位符，Index 0
        """
        super(ImageDataset, self).__init__()
        if extensions is None:
            extensions = ['jpg', 'png', 'bmp']
        if transform is None:
            transform = T.Compose([
                T.ToTensor()
            ])
        self.transform = transform
        self.files = []
        self.maxLength = maxLength
        self.placeholder = placeholder
        for ext in extensions:
            self.files.extend(glob.glob(os.path.join(path, '*.' + ext)))
        # 创建索引
        self._build_label_map()
    def __getitem__(self, item):
        """
        取样本
        :param item: ID
        :return: 图片, 标签，标签长度
        """
        img, label = self._load_file(self.files[item])
        img = self.transform(img)
        target_length = int(len(label))
        if self.maxLength is not None and target_length < self.maxLength:
            padding = torch.LongTensor([0] * (self.maxLength - target_length))
        return img, label, target_length

    def __len__(self):
        return len(self.files)

    def _build_label_map(self):
        self.label_map = []
        for file in self.files:
            label = str(os.path.basename(file).split('_')[0])
            for i in label:
                if i not in self.label_map:
                    self.label_map.append(i)
        if self.placeholder:
            self.label_map = ['_'] + sorted(self.label_map)
        else:
            self.label_map = sorted(self.label_map)

        self.label_map_length = len(self.label_map)
        self.parse_label_map = parse_label_map_c2i(self.label_map)

    def get_label_map(self):
        return ''.join(self.label_map)

    def _load_file(self, file):
        label = str(os.path.basename(file).split('_')[0])
        label = str_to_tensor(label, self.parse_label_map)
        img = np.fromfile(file, dtype=np.uint8)
        img = cv2.imdecode(img,flags=-1)
        return img, label

    def get_mean_std(self):
        mean = np.array([0, 0, 0], dtype=np.float)
        std = np.array([0, 0, 0], dtype=np.float)
        for i in tqdm(range(len(self)), desc="Evaluating Mean & Std"):
            im, _ = self._load_file(self.files[i])
            im = im.astype(np.float32) / 255.
            for j in range(3):
                mean[j] += im[:, :, j].mean()
                std[j] += im[:, :, j].std()
        mean, std = mean / len(self), std / len(self)
        return mean, std