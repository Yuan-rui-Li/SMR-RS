import numpy as np
import torch
import cv2
import os.path as osp

from RRdet.model.detectors import CascadeRCNN,SimpleMaskRCNN,MaskRCNN
from RRdet.model.backbones import ResNet
from RRdet.model.neck import FPN
from RRdet.model.dense_heads import RPNHead
from RRdet.model.bricks import *

from RRdet.apis import init_random_seed, set_random_seed
from RRdet.datasets import build_dataset, build_dataloader
from RRdet.utils import Config, get_device, compat_cfg, build_dp, get_root_logger, mkdir_or_exist
from RRdet.runner import build_runner, CosineRestartLrUpdaterHook
from RRdet.core import build_optimizer
from RRdet.apis import auto_scale_lr

path_1 = './configs/train_cfg/my_custom_config_simple.py'
path_2 = './configs/train_cfg/my_custom_config_simple_v2.py'
path_3 = './configs/train_cfg/my_custom_config_simple_mobile_v3.py'



def gene_data():

    np.random.seed(1)
    rand_array = np.random.randn(2,1080,1920,3)
    rand_tenssor = torch.from_numpy(rand_array)
    rand_tenssor = rand_tenssor.reshape(-1,3,1080,1920)
    rand_tenssor = rand_tenssor.float()

    return rand_tenssor

def load_cfg(cfg_path: str):

    cfg_dir = cfg_path
    cfg = Config.fromfile(cfg_dir)

    return cfg

def data_demo(cfg):

    data_loaders=load_data(cfg)

    #交换维度
    for _ , data_batch in enumerate(data_loaders[0]):

                img=data_batch['img'].data[0]
                img_metas=data_batch['img_metas'].data[0]
                gt_bboxes=data_batch['gt_bboxes'].data[0]
                gt_labels=data_batch['gt_labels'].data[0]
                gt_masks=data_batch['gt_masks'].data[0]

                img = img.permute(0, 2, 3, 1)
                img = img.numpy().astype(np.uint8)
                print(img_metas)

                for i in range(img.shape[0]):
                    cv2.imshow('show',img[i])
                    cv2.waitKey(500)



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
        #distributed=False
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

#resnet+fpn测试  
def train(cfg:str):

    #load config
    cfg_path =  cfg
    cfg = load_cfg(cfg_path)
    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    #seed setting
    cfg.device = get_device()
    seed = init_random_seed(None, device=cfg.device)
    set_random_seed(seed, deterministic=None)
    cfg.seed = seed

    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg_path)))

    #load data
    data_loaders=load_data(cfg)

    model_cfg = cfg.model
    model_cfg.pop('type')

    #是否禁用roi_head
    roi_head_off=False

    if roi_head_off:
        print('暂时未能启用roi_head')
        model_cfg.pop('roi_head')

    #build detector
    model = SimpleMaskRCNN(**model_cfg)
    model.init_weights()

    # put model on gpus
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, None, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=None))

    optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    #train model
    runner.run(data_loaders, cfg.workflow)


                                                                        
def main():
    train(path_1)

    return

if __name__ == '__main__':
    main()





