import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle
import numpy

import cv2
from PIL import Image
from torchvision import transforms


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""

    def __init__(self, split='train', cfg=None):
        super(S4Dataset, self).__init__()
        self.split = split
        self.cfg = cfg
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(cfg.anno_csv, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
        img_base_path = os.path.join(
            self.cfg.dir_img, self.split, category, video_name)
        audio_lm_path = os.path.join(
            self.cfg.dir_audio_log_mel, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(
            self.cfg.dir_mask, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png" % (
                video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png" % (
                video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        if self.split == 'train':
            return imgs_tensor, audio_log_mel, masks_tensor
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, category, video_name

    def __len__(self):
        return len(self.df_split)



class S4Dataset_slience(Dataset):
    """Dataset for single sound source segmentation"""

    def __init__(self, split='train', cfg=None):
        super(S4Dataset_slience, self).__init__()
        self.split = split
        self.cfg = cfg
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(cfg.anno_csv, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
        img_base_path = os.path.join(
            self.cfg.dir_img, self.split, category, video_name)
        audio_lm_path = self.cfg.dir_audio_log_mel
        mask_base_path = os.path.join(
            self.cfg.dir_mask, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png" % (
                video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png" % (
                video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        if self.split == 'train':
            return imgs_tensor, audio_log_mel, masks_tensor
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, category, video_name

    def __len__(self):
        return len(self.df_split)



import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms

random.seed(42)

class S4Dataset_ood_silence(Dataset):
    """Dataset for classification of aligned, misaligned, and silence audio-visual pairs"""
    def __init__(self, split='train', cfg=None):
        super(S4Dataset_ood_silence, self).__init__()
        self.cfg = cfg
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        # Split misaligned percentage between silence and misaligned
        self.misaligned_percentage = cfg.misaligned_percentage / 2
        self.silence_percentage = cfg.misaligned_percentage / 2
        print('misaligned_percentage (each type)', self.misaligned_percentage)

        # Load aligned data
        df_all_aligned = pd.read_csv(self.cfg.anno_csv, sep=',')
        self.df_split_aligned = df_all_aligned[df_all_aligned['split'] == split]

        # Load misaligned data (using same CSV as aligned)
        df_all_misaligned = pd.read_csv(self.cfg.anno_csv, sep=',')
        self.df_split_misaligned = df_all_misaligned[df_all_misaligned['split'] == split]

        # Load silence data (using same data structure as misaligned)
        self.df_split_silence = df_all_misaligned[df_all_misaligned['split'] == split].copy()

        print(f"{len(self.df_split_aligned)} aligned, {len(self.df_split_misaligned)} misaligned, "
              f"and {len(self.df_split_silence)} silence videos are used for {self.split}")

        # Create randomly mixed balanced indices
        self.mixed_indices = self._create_mixed_indices_with_percentage()

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _create_mixed_indices_with_percentage(self):
        num_aligned = len(self.df_split_aligned)
        num_misaligned = len(self.df_split_misaligned)
        num_silence = len(self.df_split_silence)

        # Calculate total samples and individual type counts
        total_samples = num_aligned + num_misaligned
        num_aligned_to_use = int(total_samples * (1 - self.misaligned_percentage * 2 / 100))
        num_misaligned_to_use = int(total_samples * (self.misaligned_percentage / 100))
        num_silence_to_use = total_samples - num_aligned_to_use - num_misaligned_to_use

        # Create indices for each type
        aligned_indices = list(range(num_aligned))
        misaligned_indices = list(range(num_aligned, num_aligned + num_misaligned))
        silence_indices = list(range(num_aligned + num_misaligned, 
                                   num_aligned + num_misaligned + num_silence))

        # Sample or oversample aligned indices
        if num_aligned_to_use < num_aligned:
            aligned_indices = random.sample(aligned_indices, num_aligned_to_use)
        else:
            aligned_indices = aligned_indices * (num_aligned_to_use // num_aligned) + \
                            random.sample(aligned_indices, num_aligned_to_use % num_aligned)

        # Sample or oversample misaligned indices
        if num_misaligned_to_use < num_misaligned:
            misaligned_indices = random.sample(misaligned_indices, num_misaligned_to_use)
        else:
            misaligned_indices = misaligned_indices * (num_misaligned_to_use // num_misaligned) + \
                                random.sample(misaligned_indices, num_misaligned_to_use % num_misaligned)

        # Sample or oversample silence indices
        if num_silence_to_use < num_silence:
            silence_indices = random.sample(silence_indices, num_silence_to_use)
        else:
            silence_indices = silence_indices * (num_silence_to_use // num_silence) + \
                            random.sample(silence_indices, num_silence_to_use % num_silence)

        # Combine and shuffle all indices
        all_indices = aligned_indices + misaligned_indices + silence_indices
        random.shuffle(all_indices)

        return all_indices

    def update_misaligned_percentage(self, new_percentage):
        self.misaligned_percentage = new_percentage / 2  # Split between silence and misaligned
        self.silence_percentage = new_percentage / 2
        self.mixed_indices = self._create_mixed_indices_with_percentage()
        print(f"Updated misaligned percentage to {new_percentage}% (each type: {new_percentage/2}%)")

    def __getitem__(self, index):
        # Use the mixed balanced index
        mixed_index = self.mixed_indices[index]
        num_aligned = len(self.df_split_aligned)
        num_misaligned = len(self.df_split_misaligned)
        
        # Determine the type of sample and get corresponding data
        if mixed_index < num_aligned:
            # Aligned sample
            df_one_video = self.df_split_aligned.iloc[mixed_index]
            video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
            img_base_path = os.path.join(self.cfg.dir_img, self.split, category, video_name)
            audio_lm_path = os.path.join(self.cfg.dir_audio_log_mel, self.split, category, video_name + '.pkl')
            mask_base_path = os.path.join(self.cfg.dir_mask, self.split, category, video_name)
            is_aligned = True
        elif mixed_index < num_aligned + num_misaligned:
            # Misaligned sample
            df_one_video = self.df_split_misaligned.iloc[mixed_index - num_aligned]
            video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
            img_base_path = os.path.join(self.cfg.dir_img, self.split, category, video_name)
            audio_lm_path = os.path.join(self.cfg.mis_dir_audio_log_mel, self.split, category, video_name + '.pkl')
            mask_base_path = 'empty_GT.png'
            is_aligned = False
        else:
            # Silence sample
            df_one_video = self.df_split_silence.iloc[mixed_index - num_aligned - num_misaligned]
            video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
            img_base_path = os.path.join(self.cfg.dir_img, self.split, category, video_name)
            audio_lm_path = self.cfg.silence_dir_audio_log_mel
            mask_base_path = 'empty_GT.png'
            is_aligned = False

        audio_log_mel = load_audio_lm(audio_lm_path)
        
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, f"{video_name}_{img_id}.png"), 
                                            transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            if is_aligned:
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, f"{video_name}_{mask_id}.png"), 
                                                 transform=self.mask_transform, mode='1')
            else:
                mask = load_image_in_PIL_to_Tensor(mask_base_path, transform=self.mask_transform, mode='1')
            masks.append(mask)
        
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        label = torch.tensor(1 if is_aligned else 0, dtype=torch.float32)

        if self.split == 'train':
            return imgs_tensor, audio_log_mel, masks_tensor, label
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, label, category, video_name

    def __len__(self):
        return len(self.mixed_indices)