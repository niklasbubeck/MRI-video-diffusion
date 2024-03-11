import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom
import cv2
import pickle

from typing import Union
import lmdb

import torchvision
from torchvision.transforms import functional as F

MEAN_SAX_LV_VALUE = 222.7909
MAX_SAX_VALUE = 487.0
MEAN_4CH_LV_VALUE = 224.8285
MAX_4CH_LV_VALUE = 473.0

def get_torchvision_transforms(cfg, mode):
    assert mode in {'train', 'test'}
    if mode == 'train':
        transforms_cfg = cfg.dataset.train.transforms
    else:
        transforms_cfg = cfg.dataset.test.transforms

    transforms = []
    for t in transforms_cfg:
        if hasattr(torchvision.transforms, t['name']):
            transform_cls = getattr(torchvision.transforms, t['name'])(**t['params'])
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms

def normalize_image_with_mean_lv_value(im: Union[np.ndarray, torch.Tensor], mean_value=MEAN_SAX_LV_VALUE, target_value=0.5) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize such that LV pool has value of 0.5. Assumes min value is 0.0. """
    im = im / (mean_value / target_value)
    im = im.clip(min=0.0, max=1.0)
    return im

class UKBB_lmdb(Dataset):
    def __init__(self,
                 config,
                 path=os.path.expanduser('/vol/aimspace/users/bubeckn/diffae/datasets/ukbb_MedMAE.lmdb'),
                 ):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.config = config
        self.slice_res = 8
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length
    

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:

            key = f'{str(index).zfill(5)}'.encode(
            'utf-8')
            temp = txn.get(key)
            npz = pickle.loads(temp)
            sa = npz['sa']
            # subject = npz['subject']
            sa = torch.from_numpy(sa).unsqueeze(0).type(torch.float)         # c, h ,w

            # sa = torch.repeat_interleave(sa, 3, dim=0)
            # if self.config.dataset.normalize:
            sa = sa * 2 - 1
            print(sa.shape)
        return sa, index


class UKBBPartial(Dataset):

    def __init__(self, config, restrict_data=False) -> None:
        """
        Constructor Method
        """

        extensions = ['nii', 'gz', "nii.gz"]

        self.target_slices = config.dataset.get("slice_res", 64)
        self.target_time_res = config.dataset.get("time_res")
        self.target_resolution = config.dataset.get("res", 256)
        self.root_dir = config.dataset.get("data_path", 256)

        self.input_fnames = []
        self.la_fnames =[]
        # self.meta_fnames = []
        for ext in extensions:
            try:
                    self.input_fnames += sorted(glob.glob(f'{self.root_dir}/**/*sa_cropped.{ext}', recursive=True))
                    # self.la_fnames += sorted(glob.glob(f'{self.root_dir}/**/*la_2ch.{ext}', recursive=True))
                    # self.meta_fnames += sorted(glob.glob(f'{self.root_dir}/**/Info.cfg', recursive=True))
            except:
                ImportError('No data found in the given path')

        if restrict_data:
            length = int(len(self.meta_fnames)*restrict_data)
            # self.meta_fnames = self.meta_fnames[0:length]
            self.input_fnames = self.input_fnames[0:length]
        print(f'{len(self.input_fnames)} files found in {self.root_dir}')
        # print(f'{len(self.la_fnames)} files found in {self.root_dir}/{folder}')
        # assert len(self.input_fnames) == len(self.la_fnames), f"number of sa and la input not equal"
        assert len(self.input_fnames) != 0, f"Given directory contains 0 images. Please check on the given root: {self.root_dir}"


    @property
    def fnames(self):
        return self.targets_fnames

    
    def load_nifti(self, fname:str):
        nii = nib.load(fname).get_fdata()
        return nii

    def load_meta_patient(self, fname:str):

        file = open(fname, 'r')
        content = file.read()
        config_dict = {}
        lines = content.split("\n") #split it into lines
        for path in lines:
            split = path.split(": ")
            if len(split) != 2: 
                break
            key, value = split[0], split[1]
            config_dict[key] = value

        return config_dict

    def preprocess(self, meta:dict):
        group = meta['Group']
        weight = float(meta['Weight'])
        height = float(meta['Height'])

        weight = weight / 100
        height = height / 200

        if group == "DCM":
            group = 1
        elif group == "HCM":
            group = 2
        elif group == "MINF":
            group = 3
        elif group == "ARV":
            group = 4
        elif group == "LV":
            group = 5
        elif group == "RV":
            group = 6
        elif group == "NOR":
            group = 7
        else: 
            print("Group not known!!")

        return group/7, height, weight

    def __len__(self):
        return len(self.input_fnames)

    def __getitem__(self, idx):
        # load the target image
        nii = self.load_nifti(self.input_fnames[idx]) # h, w, s, t
        cond_idx = np.random.randint(0, self.target_slices)
        
        # # Resample volume to target depth/slices
        # h, w, s_orig, t_orig = nii.shape
        # nii = zoom(nii, (1, 1, self.target_slices/s_orig, self.target_time_res/t_orig), order=3)

        # use random slice 2d+t as reference
        nii_t = nii[...,cond_idx,0:self.target_time_res] # h, w, t

        # use 5th slice as conditioning
        nii_mid = nii[...,5,:] # h, w, t 
        

        # # Load LA image 
        # la = self.load_nifti(self.la_fnames[idx])
        # la_o_h, la_o_w, la_o_s, la_o_t = la.shape

        # # Crop LA image 
        # left = (w - self.target_resolution) // 2
        # top = (h - self.target_resolution) // 2
        # right = left + self.target_resolution
        # bottom = top + self.target_resolution
        # la = la[top:bottom, left:right, 0, 0]

        # # Error handling for la images smaller than 128
        # la_h, la_w = la.shape
        # if la_h != self.target_resolution or la_w != self.target_resolution:
        #     print(f"Weird stuff: {la_o_h} {la_o_w} --> {la_h} {la_w}")
        #     return self.__getitem__(idx + 1)
        
        # # Pad Video to a common length
        # diff_h, diff_w = abs(h - self.target_resolution), abs(w - self.target_resolution)
        # nii = np.pad(nii, [(diff_h//2, diff_h//2),(diff_w//2, diff_w//2), (0,0)], mode='constant', constant_values=0) # only works for even resolutions
        
        
        nii_t = torch.tensor(nii_t).permute((2,0,1)).unsqueeze(0).type(torch.float) # b, t, h, w
        nii_mid = torch.tensor(nii_mid).permute((2,0,1)).unsqueeze(0).type(torch.float) # b, t, h, w

        # Add conditional text embeds
        slices_coord = torch.tensor(cond_idx)[None,None,...].type(torch.float) # convert to tensor and add batch and channel dimensions

        
        # Add conditional image
        ed_sa = nii_mid[:, 0, :, :] # take ED first frame/ b, h, w
        # ed_la = torch.tensor(la).unsqueeze(0).type(torch.float)
        # cond_frame = torch.cat((ed_sa, ed_la))

        # # Debug area
        # print("slices_coord: ", slices_coord)
        # print("cond_frame: ", cond_frame)
        # print("nii: ", nii.max())
        # print(nii)
        # nii = nii / 255
        # plt.imsave('/home/niklas/Desktop/videos/cond5frame.png', ed_sa[0], cmap='grey')
        # plt.imsave('/home/niklas/Desktop/videos/ref1frame.png', nii_t[0, 0, :, :], cmap='grey')


        return nii_t, ed_sa ,slices_coord, self.input_fnames[idx]
    