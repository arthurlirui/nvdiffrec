# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json

import torch
import numpy as np

from render import util

from .dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path):
    #filename = os.path.basename(path).split('.')[0]
    files = glob.glob(path + '.*')
    if len(files) == 0:
        files = glob.glob(path)
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetNERF(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy   = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv     = mv @ util.rotate_x(-np.pi / 2)
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img      = []
        fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }


class DatasetIRON(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))

        # Load xml file from mitsuba2 setup
        #self.cfg_mitsuba2_xml = json.load(open(XXX, 'r'))

        self.n_images = len(self.cfg)
        filelist = list(self.cfg.keys())

        # Determine resolution & aspect ratio
        #self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        #self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.resolution = self.cfg[filelist[0]]['img_size']
        self.aspect = self.resolution[1] / self.resolution[0]

        fname = list(self.cfg.keys())[0]
        focal = self.cfg[fname]['K'][0, 0]
        self.focal = focal
        width, height = self.resolution
        fov = np.rad2deg(np.arctan(width / 2.0 / focal) * 2.0)
        fovx = fov
        fovy = util.fovx_to_fovy(fovx, self.aspect)
        self.fovx = fovx
        self.fovy = fovy

        if self.FLAGS.local_rank == 0:
            print("DatasetIRON: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            #for i in range(self.n_images):
            #    self.preloaded_data += [self._parse_frame(self.cfg, i)]
            for i, fname in enumerate(filelist):
                self.preloaded_data += [self._parse_frame(self.cfg, fname)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fname = list(cfg.keys())[idx]
        # load xml file here
        #focal = cfg[fname]['K'][0, 0]
        #width, height = self.resolution
        #fov = np.rad2deg(np.arctan(width / 2.0 / focal) * 2.0)
        #fovx = fov
        #fovy = util.fovx_to_fovy(fovx,self.aspect)
        #fovy = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
        proj = util.perspective(self.fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        #img = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
        img = _load_img(os.path.join(self.base_dir, fname))
        #mv = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv = torch.linalg.inv(torch.tensor(cfg[fname]['W2C']))
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...]  # Add batch dimension

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res

        img = []
        #fovy = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)
        fovy = self.fovy

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv': mv,
            'mvp': mvp,
            'campos': campos,
            'resolution': iter_res,
            'spp': self.FLAGS.spp,
            'img': img
        }