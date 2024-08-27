import numpy as np
import torch
import cv2
import time
from PIL import Image
import copy
import os
import datetime

from RRdet.user_utils import (get_bbox_coordinate, show_masks_bboxes, load_cfg,
                                gene_data, load_data, show_bboxes, bbox2pad, 
                                mask_blend, show_mask_singal, attach_bbox,
                                new_path,get_allocated_memory)

from RRdet.model.builder import build_detector

from RRdet.apis import init_random_seed, set_random_seed

from RRdet.utils import get_device, compat_cfg, build_dp, get_root_logger
from RRdet.apis import auto_scale_lr, init_detector, inference_detector

          
colors = [(50, 255, 50), (10, 10, 255), (255,10,10), (10, 10, 255), (65, 105, 225), (135, 206, 235),
                  (162, 205, 90), (255, 246, 143), (238, 238, 209), (188, 143, 143), (222, 184, 135), (205, 85, 85)]


def get_infer_results(cfg:str, checkpoint_file:str, img_file:str, proposals=None):
    #load config 
    cfg = load_cfg(cfg)
    cfg = compat_cfg(cfg)
    model_cfg = cfg.model
    
    model = init_detector(cfg, checkpoint_file,device='cuda:0')
    mem_mb = get_allocated_memory('cuda:0')
    print(f'memory:{mem_mb}')
    for _ in range(1):

        rusults, t_dict = inference_detector(model, img_file, proposals)

        time.sleep(0.5)
    mem_mb = get_allocated_memory('cuda:0')
    print(f'memory:{mem_mb}')
    return rusults, model


def show_results(cfg:dict,
                 use_custom_proposal=False, 
                 merge_masks=True, 
                 merge_bboxes=True):
    '''
    @breif:推理模型并展示结果,包括masks、bboxes
    @note:输入仅限单张图片
    '''
    path = cfg
    cfg_file, checkpoint_file, img_file = path['cfg_file'],path['checkpoint_file'],path['img_file']
    img = cv2.imread(img_file)
    
    #图片保存路径
    base_path = '/home/rui/桌面/RRdetection/custom_data/compare/'
    mask_save_path = base_path+'simple/'+os.path.basename(img_file)
    #是否使用rpn_head提供的proposals
    if use_custom_proposal:
        bbox = get_bbox_coordinate(img)
        proposal = bbox
        proposals = [[torch.tensor([proposal,],dtype=torch.float32).view(-1,4)]]
        infer_results, _ = get_infer_results(cfg_file,checkpoint_file,img_file,proposals)
    else:
        infer_results, _ = get_infer_results(cfg_file,checkpoint_file,img_file)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    '''
    get masks:
    masks:list(list([tensor,...],...),...)
    [0][0]代表第0个图片中的第0类物体的mask
    masks:np.ndarray
    '''
    masks = infer_results['masks'][0][0]
    num_masks = len(masks)
    print(num_masks)

    '''
    get proposals:
    proposals:list[tensor[tl_x, tl_y, br_x, br_y, score, ...],...]
    proposals为rpn_head阶段的直接输出结果
    [0]表示输入的第0张图片
    '''
    proposals = infer_results['proposals'][0]
    if proposals is not None:
        num_proposals = len(proposals)
        print(num_proposals)

    '''
    get bboxes:
    bboxes:tensor[tl_x, tl_y, br_x, br_y,...]
    bboxes为实际上用于生成masks的bboxes,经过nms操作后其数量小于等于输入的proposals
    [0]表示输入的第0张图片,[0]表示第0类(不包括背景)
    '''
    bboxes = infer_results['bboxes'][0][0]
    num_bboxes = len(bboxes)
    print(num_bboxes)

    '''
    get bboxes_score:
    bboxes_score:tensor[score,...]
    bboxes_score为bboxes的预测得分,即在proposals中对应的bboxes的得分
    '''
    if not use_custom_proposal:
        bbox_scores = infer_results['bboxes_score']
    else:
        bbox_scores = torch.Tensor([1])
    num_bboxes_score = len(bbox_scores) if bbox_scores is not None else 0


    assert num_bboxes == num_bboxes_score == num_masks
    
    if num_masks != 0 and num_bboxes != 0 :
        show_masks_bboxes(img,
                    masks,
                    bboxes,
                    bbox_scores,
                    num_masks,
                    num_bboxes,
                    merge_masks,
                    merge_bboxes,
                    bbox_line_width=12,
                    show_blend_result=True,
                    blend_bboxes_masks=True,
                    show_bboxes=True,
                    bbox_color=colors,
                    show_text=False,
                    mask_save_path=mask_save_path
                    )


def show_anchors_proposals(cfg:dict,
                            save_paths:dict,
                            show_all_anchors:bool=False):
    '''
    @breif:rpn阶段生成的anchors至proposals的变化对比
    '''
    path = cfg
    cfg_file, checkpoint_file, img_file = path['cfg_file'],path['checkpoint_file'],path['img_file'] 
    results, model = get_infer_results(cfg_file,checkpoint_file,img_file)
    #rpn输出的提议框对应的锚框
    anchors = model.rpn_head.keep_anchors_list
    #rpn输出的的proposals，不包含score
    proposals = model.rpn_head.keep_proposal_list
    #单张图片的proposals的score
    scores = model.rpn_head.keep_proposal_score
    #得到掩膜
    masks = results['masks'][0][0]
    #被保留的锚框和提议框的编号
    keep = results['keep'].cpu().numpy()
    #rpn阶段输出的提议框和对应锚框的数量
    num_anchors = len(anchors[0])
    num_proposals = len(proposals[0])
    assert num_anchors == num_proposals
    bboxes=anchors[0].cpu().numpy()

    #将BGR格式转换为RGB格式
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_color = (192,192,192)
    img = cv2.copyMakeBorder(image.copy(), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=pad_color)

    #图片保存路径
    anchors_path = save_paths['anchors_path']
    proposals_path = save_paths['proposals_path']
    keep_anchors_path = save_paths['keep_anchors_path']
    merge_proposals_path = save_paths['merge_proposals_path']
    proposals_masks_path = save_paths['proposals_masks_path']
    blend_path = save_paths['blend_path']
    if show_all_anchors:
        #显示rpn输出的所有proposals对应的anchors
        img, bboxes = bbox2pad(bboxes, image, pad_color=pad_color)
        # show_bboxes(bboxes, 
        #             img,
        #             merge_bboxes=True,
        #             text='anchor',
        #             show_text=False,
        #             bbox_line_width=1,
        #             save_path=anchors_path,
        #             show_coners=False)
        # print(len(masks))
        bboxes=proposals[0].cpu().numpy()
        img, bboxes = bbox2pad(bboxes, image, pad_color=pad_color)
        show_bboxes(bboxes, 
                    img,
                    merge_bboxes=True,
                    text='anchor',
                    show_text=False,
                    bbox_line_width=6,
                    save_path=proposals_path,
                    show_coners=False)
                    
    
    if not show_all_anchors:
        #显示在roi_head阶段保留下来的proposals对应的anchors
        bboxes=anchors[0][keep].cpu().numpy()
        #padding
        img, bboxes = bbox2pad(bboxes, image, pad_color=pad_color)
        show_bboxes(bboxes, 
                    img,
                    merge_bboxes=True,
                    show_text=False,
                    text='anchor-keep',
                    bbox_line_width=12,
                    save_path=keep_anchors_path,
                    bbox_color = colors)

        #显示合并后的proposals
        bboxes = results['bboxes'][0][0].cpu().numpy()
        #padding
        img, bboxes = bbox2pad(bboxes, image, pad_color=pad_color)
        show_bboxes(bboxes, 
                    img,
                    merge_bboxes=True,
                    show_text=False,
                    text='proposals',
                    bbox_line_width=12,
                    save_path=merge_proposals_path,
                    bbox_color = colors)

        
        #显示mask和相应的proposals
        bboxes = results['bboxes'][0][0].cpu().numpy()
        assert len(bboxes) == len(masks)
        blend = zip(bboxes, masks)
        for i, ele in enumerate(blend):
            bbox = ele[0]
            mask = ele[1].cpu().numpy()
            img_blend = mask_blend(image.copy(), mask, color=colors[i])
            if i == 0:
                last_blend=img_blend
            if i>0:
                last_blend=mask_blend(last_blend, mask, color=colors[i])
            img_blend = cv2.copyMakeBorder(img_blend, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=pad_color)
            attach_bbox(img_blend,
                        bbox+30,
                        custom_text='proposal',
                        show_text=False,
                        bbox_color=colors[i],
                        bbox_line_width=12)
            img_blend = Image.fromarray(img_blend)
            img_blend.show()
            if proposals_masks_path is not None:
                path = new_path(proposals_masks_path, i)
                img_blend.save(path)
        if blend_path is not None:
            last_blend = Image.fromarray(last_blend)
            last_blend.save(blend_path)

        # bboxes = torch.tensor(bboxes)
        # #显示最终掩膜
        # masks_sum=copy.deepcopy(masks[0])
        # for i in range(len(masks)-1):
        #     masks_sum|=masks[i+1]
        # mask = masks_sum.cpu().numpy()
        # img_blend = mask_blend(image.copy(), mask, alpha=0.3, color=(249,125,28))
        # img_blend = Image.fromarray(img_blend)
        # img_blend.show()
        # if blend_path is not None:
        #         img_blend.save(blend_path)

        #显示最终掩膜与边框
        #显示合并后的proposals
        bboxes = results['bboxes'][0][0].cpu().numpy()
        #padding
        img, bboxes = bbox2pad(bboxes, np.array(last_blend), pad_color=pad_color)
        show_bboxes(bboxes, 
                    img,
                    merge_bboxes=True,
                    show_text=False,
                    text='proposals',
                    bbox_line_width=12,
                    save_path=proposals_masks_path,
                    bbox_color = colors)

    


def data_demo(cfg:dict, save_result=False):
    import os
    from RRdet.user_utils import save_eval

    cfg_file = cfg['cfg_file']
    cfg_file = load_cfg(cfg_file)

    #seed setting
    cfg_file.device = get_device()
    seed = init_random_seed(None, device=cfg_file.device)
    set_random_seed(seed, deterministic=None)
    cfg_file.seed = seed

    data_loaders=load_data(cfg_file)

    #交换维度
    gt_meta_dict={}
    for _ , data_batch in enumerate(data_loaders[0]):
        '''
        data_batch包含一组图片的信息,图片的数量由samples_per_gpu决定
        '''
        # print(data_batch)
        # img=data_batch['img'].data[0]
        img_metas=data_batch['img_metas'].data[0]
        gt_bboxes=data_batch['gt_bboxes'].data[0]
        # gt_labels=data_batch['gt_labels'].data[0]
        # gt_masks=data_batch['gt_masks'].data[0]

        # img = img.permute(0, 2, 3, 1)
        # img = img.numpy().astype(np.uint8)
        # print(img_metas)
        #0代表第0张图片
        #由于eval无法读取np.array格式的数据,故先将数据转换为list格式,待读取出数据后再转换为np.array格式。
        gt_bbox = gt_bboxes[0].cpu().numpy().astype(np.int16).tolist()
        file_name = os.path.basename(img_metas[0]['filename']) 

        gt_meta_dict[file_name] = gt_bbox
        # for i in range(img.shape[0]):
        #     print(gt_bboxes)
        #     cv2.imshow('show',img[i])
        #     cv2.waitKey(500)
    if save_result:
        save_eval(gt_meta_dict,'gt_metas.txt')


def backbone_neck_Demo(cfg:dict):

    #load config 
    cfg_file = cfg['cfg_file']
    cfg_file = load_cfg(cfg_file)
    cfg_file = compat_cfg(cfg_file)
    logger = get_root_logger(log_level=cfg_file.log_level)

    #seed setting
    cfg_file.device = get_device()
    seed = init_random_seed(None, device=cfg_file.device)
    set_random_seed(seed, deterministic=None)
    cfg_file.seed = seed

    #load data
    data_loaders=load_data(cfg_file)

    model_cfg = cfg_file.model
    model_type = model_cfg.pop('type')
    #暂时未能启用'roi_head'
    model_cfg.pop('roi_head')

    #build detector
    model = build_detector(model_cfg)
    model.init_weights()

    backbone_out = model.backbone(gene_data())
    neck_out = model.neck(backbone_out)

    #print results
    print('backbone part')
    print(len(backbone_out))
    for i in range(len(backbone_out)):
        print(backbone_out[i].size())
   
    print('neck part')
    print(len(neck_out))
    for i in range(len(neck_out)):
        print(neck_out[i].size())

    return

                                                                           
def main():
    #保存路径,只需修改base_path
    base_path = './demo'
    folder_name = 'Anchors_And_Proposals_'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_paths = dict(
    anchors_path = os.path.join(base_path,folder_name,'multi_anchors.jpg'),
    proposals_path = os.path.join(base_path,folder_name,'multi_proposals.jpg'),
    keep_anchors_path = os.path.join(base_path,folder_name,'keep_anchors.jpg'),
    merge_proposals_path = os.path.join(base_path,folder_name,'proposals.jpg'),
    proposals_masks_path = os.path.join(base_path,folder_name,'proposal_mask_path.jpg'),
    blend_path = os.path.join(base_path,folder_name,'mask_blend.jpg')
    )
    #要推理的图片
    img_file = './demo/test/00.jpg'
    #设置权重和配置文件的保存路径
    #SMR-RS
    cfg_1 = dict(cfg_file = './configs/infer_cfg/SMR/my_custom_config_simple_18_3.py',
                checkpoint_file = './train_results/simple/epoch_18_3.pth',
                img_file = img_file)
    #SMR-MN3
    cfg_2 = dict(cfg_file = './configs/infer_cfg/SMR/my_custom_config_simple_23.py',
                checkpoint_file = './train_results/simple/epoch_23.pth',
                img_file = img_file)
    #SMR-RS-MN3
    cfg_3 = dict(cfg_file = './configs/infer_cfg/SMR/my_custom_config_simple_22.py',
                checkpoint_file = './train_results/simple/epoch_22.pth',
                img_file = img_file)
    #SMR-RS_v2
    cfg_4 = dict(cfg_file = './configs/infer_cfg/SMRv2/my_custom_config_simple_v2_01.py',
                checkpoint_file = './train_results/simple_v2/epoch_01.pth',
                img_file = img_file) 


    torch.cuda.empty_cache()  # 释放显存
    '''
    本文件中三个主要函数的作用:
    @backbone_neck_Demo
        展示骨干网络(backbone)和颈部网络(neck)输出数据的尺寸大小
    @show_results
        推理模型并展示结果,包括masks、bboxes
    @rpn_anchor_proposal_compare
        展示推理过程中锚框(anchors)到提议框(proposals)的变化
    @data_demo
        加载数据集并获取数据集中的指定信息,具体请查看代码
    '''
    #data_demo(cfg=cfg_2)
    #backbone_neck_Demo(cfg = cfg_2)
    show_anchors_proposals(cfg = cfg_2, save_paths = save_paths, show_all_anchors = False)
    return

if __name__ == '__main__':
    main()





