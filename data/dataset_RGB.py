import os
import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


# 🔥 新增：无参考数据集
class NonRefDataReader(Dataset):
    """
    只输入 input 文件夹，不需要 target
    返回： inp_img, filename
    """
    def __init__(self, img_dir, inp='input', mode='train', ori=False, img_options=None):
        super(NonRefDataReader, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, inp)))
        self.inp_filenames = [os.path.join(img_dir, inp, x) for x in inp_files if is_image_file(x)]

        self.mode = mode
        self.img_options = img_options
        self.sizex = len(self.inp_filenames)

        # 数据增强
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Flip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop(height=img_options['h'], width=img_options['w']),
            ])
        else:
            if ori:
                self.transform = A.Compose([A.NoOp()])
            else:
                self.transform = A.Compose([A.Resize(height=img_options['h'], width=img_options['w'])])

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.inp_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        inp_img = np.array(inp_img)

        transformed = self.transform(image=inp_img)
        inp_img = F.to_tensor(transformed['image'])

        filename = os.path.basename(inp_path)
        return inp_img, filename


# 🔥 原版：有参考数据集
class DataReader(Dataset):
    """
    输入 input + target 文件夹
    返回： inp_img, tar_img, filename
    """
    def __init__(self, img_dir, inp='input', tar='target', mode='train', ori=False, img_options=None):
        super(DataReader, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, inp)))
        self.inp_filenames = [os.path.join(img_dir, inp, x) for x in inp_files if is_image_file(x)]

        if tar is None or tar == "":
            # 🔥 无参考模式时，直接用 None 占位
            self.tar_filenames = [None] * len(self.inp_filenames)
        else:
            tar_files = sorted(os.listdir(os.path.join(img_dir, tar)))
            self.tar_filenames = [os.path.join(img_dir, tar, x) for x in tar_files if is_image_file(x)]

        self.mode = mode
        self.img_options = img_options
        self.sizex = len(self.inp_filenames)

        # 数据增强
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Flip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop(height=img_options['h'], width=img_options['w']),
            ], additional_targets={'target': 'image'})
        else:
            if ori:
                self.transform = A.Compose([A.NoOp()], additional_targets={'target': 'image'})
            else:
                self.transform = A.Compose([A.Resize(height=img_options['h'], width=img_options['w'])],
                                           additional_targets={'target': 'image'})

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        inp_img = np.array(inp_img)

        if tar_path is not None:
            tar_img = Image.open(tar_path).convert('RGB')
            tar_img = np.array(tar_img)
            transformed = self.transform(image=inp_img, target=tar_img)

            inp_img = F.to_tensor(transformed['image'])
            tar_img = F.to_tensor(transformed['target'])
            filename = os.path.basename(tar_path)
            return inp_img, tar_img, filename
        else:
            # 🔥 无参考时，只返回 inp
            transformed = self.transform(image=inp_img)
            inp_img = F.to_tensor(transformed['image'])
            filename = os.path.basename(inp_path)
            return inp_img, filename

