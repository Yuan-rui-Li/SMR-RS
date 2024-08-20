import numpy as np
import torch
import pandas as pd
import csv
import json
import pickle
import os
from typing import Tuple

from ..datasets import build_dataset, build_dataloader
from .cfg import load_cfg
from ..apis import init_random_seed, set_random_seed
from ..utils import get_device

def new_path(path:str, i:int):
    '''
    不改变扩展名，将指定编号加在文件名后上
    '''
    file_name_split = os.path.splitext(path)
    #得到除扩展名以外的部分
    base_name = file_name_split[0]
    #得到扩展名
    file_extension = file_name_split[1]
    #新文件名
    new_path = base_name+str(i)+file_extension

    return new_path

def save_eval(data:Tuple[str,dict],file_name:str):
    '''
    @brief:使用eval来保存文件
    @args:
        data(str):A str has expression format inside like ''dict(a=1,b=2)''.
        file_name(str): Absolutely file-direction of data.
    '''
    # if isinstance(data, str):
    #     data = eval(data)
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(str(data))  # dict to str
    


def load_eval(file_name:str):
    '''
    @brief:Open a file by eval
    @args:
        file_name(str): Absolutely file-direction of data.
    '''
    with open(file_name, 'r', encoding='utf-8') as f:
        data = eval(f.read())  # eval
        return data



def save_pickle(data,file_name:str):
    '''
    @brief:使用pickle来保存文件
    @args:
        data:Any data that you want to save as piclkle format,it could be str、array、dict etc.
        file_name(str): Absolutely file-direction of data.
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name:str):
    '''
    @brief:Open file by pickle
    @args:
        file_name(str): Absolutely file-direction of data.
    '''
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def gene_data():

    np.random.seed(1)
    rand_array = np.random.randn(2,1080,1920,3)
    rand_tensor = torch.from_numpy(rand_array)
    rand_tensor = rand_tensor.reshape(-1,3,1080,1920)
    rand_tensor = rand_tensor.float()

    return rand_tensor


    #datasets加载数据集测试
def load_data(cfg):

    #loading datasets from file then buide datasets object
    datasets = [build_dataset(cfg.data.train)]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    train_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
 
    }
    #after sampler setting then using torch.utils.data.Dataloder to create data_loaders
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in datasets]

    return data_loaders

def datasets_out(cfg:str, return_img=False):
    '''
    @breif: Export a datasets, make it could be visible.
        It contain "img", "img_meta", "proposals", "gt_bboxes",
        "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

        The "img_meta" item is always populated.  The contents of the "img_meta"
        dictionary depends on "meta_keys". By default this includes:

            - "img_shape": shape of the image input to the network as a tuple \
                (h, w, c).  Note that images may be zero padded on the \
                bottom/right if the batch tensor is larger than this shape.

            - "scale_factor": a float indicating the preprocessing scale

            - "flip": a boolean indicating if image flip transform was used

            - "filename": path to the image file

            - "ori_shape": original shape of the image as a tuple (h, w, c)

            - "pad_shape": image shape after padding

            - "img_norm_cfg": a dict of normalization information:

                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
    @args:
        cfg:Config file path.
    @return:
        data_batch_list(list):List contain all data, it's element is a dict contain one batch.
    '''

    # cfg_path = './configs/data_config.py'
    cfg = load_cfg(cfg)

    #seed setting
    cfg.device = get_device()
    seed = init_random_seed(None, device=cfg.device)
    set_random_seed(seed, deterministic=None)
    cfg.seed = seed

    data_loaders=load_data(cfg)

    #交换维度
    data_batch_list=[]
    for _ , data_batch in enumerate(data_loaders[0]):
        '''
        data_batch包含一组图片的信息,图片的数量由samples_per_gpu决定
        '''
        data_batch_dict={}
        if return_img:
            img=data_batch['img'].data[0]
        img_metas_list=data_batch['img_metas'].data[0]
        gt_bboxes_list=data_batch['gt_bboxes'].data[0]
        gt_labels_list=data_batch['gt_labels'].data[0]
        gt_masks_list=data_batch['gt_masks'].data[0]

        data_batch_dict['img_metas'] = img_metas_list
        data_batch_dict['gt_bboxes'] = gt_bboxes_list
        data_batch_dict['gt_labels'] = gt_labels_list
        data_batch_dict['gt_masks'] = gt_masks_list

        data_batch_list.append(data_batch_dict)

    return data_batch_list

        # img = img.permute(0, 2, 3, 1)
        # img = img.numpy().astype(np.uint8)
        # print(img_metas)
        #0代表第0张图片
        #由于eval无法读取np.array格式的数据,故先将数据转换为list格式,待读取出数据后再转换为np.array格式。
        # gt_bbox = gt_bboxes[0].cpu().numpy().astype(np.int16).tolist()

        # for i in range(img.shape[0]):
        #     print(gt_bboxes)
        #     cv2.imshow('show',img[i])
        #     cv2.waitKey(500)

