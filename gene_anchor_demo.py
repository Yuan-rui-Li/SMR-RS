import cv2
import numpy as np
from PIL import Image
import torch

from RRdet.user_utils import show_bboxes
from RRdet.core.anchor import AnchorGenerator,anchor_inside_flags
from RRdet.user_utils import random_choice

colors = [(50, 255, 50), (10, 10, 255), (255,10,10), (10, 10, 255), (65, 105, 225), (135, 206, 235),
                  (162, 205, 90), (255, 246, 143), (238, 238, 209), (188, 143, 143), (222, 184, 135), (205, 85, 85)]

def gene_anchors(imgDir:str,
                 image_heigt:int,
                 image_width:int,
                 scales:list[int],
                 ratios:list[int],
                 strides:list[int],
                 num_choices:int=10,
                 featmap_sizes:list = [(270, 480),(135, 24),(68, 120),(34, 60)]):
    
    cfg = dict( scales=scales,
                ratios=ratios,
                strides=strides,#不同featuremap上bbox的间距,其倒数为featuremap相较于原图的比例
                use_custom_method_gene_anchors=False
                )
    '''
    cfg的一些不常用的设置:
    # base_sizes=[10,10,10,10],#不同featuremap上bbox基本尺寸,其倒数为featuremap相较于原图的比例
    # base_height=[19,19,19,19],#每个featuremap上bbox基本高度
    # base_width=[5.4,5.4,7.5,10.8],#每个featuremap上bbox基本宽度
    # centers = [(4,4),(8,8),(16,16),(32,32)],

    ---------''base_anchors''-------------
    @brief:base_anchors是在原图上每个位置生成的anchor组,anchor即bbox
    list(tensor([[tl_x,tl_y,br_x,br_y],...num_strides*num_ratios],...num_levels)],num_levels为featuremap的个数
    @note:

    base_anchors中boxes的个数为num_strides*num_ratios*num_featuremaps
    anchor的高h_ratios = torch.sqrt(ratios)
    anchor的宽w_ratios = 1 / h_ratios

    anchor高和宽的计算方法:
    1.如果没有指定base_sizes:
    >>>for stride in strides:
    >>>    for scale in scales:
    >>>        for ratio in ratios:
    >>>            anchor_height = stride*h_ratio*scale
    >>>            anchor_width = stride*w_ratio*scale
    '''
    gene_anchor = AnchorGenerator(**cfg)
    base_anchors = gene_anchor.base_anchors
    print(f"基础锚框的大小为:{base_anchors}")

    #假设的参数
    

    #list(tensor,...num_featuremaps)
    multi_level_anchors = gene_anchor.grid_priors(featmap_sizes, device='cpu')
    num_multi_level_anchors = sum([len(single_level_anchors)for single_level_anchors in multi_level_anchors])
    #list(tensor,...num_featuremaps)
    #获取中心在图像内部的anchor的flag
    multi_level_valid_flags = gene_anchor.valid_flags(featmap_sizes, (image_heigt,image_width), device='cpu')
    num_multi_level_valid_flags = sum([len(single_level_valid_flags)for single_level_valid_flags in multi_level_valid_flags])

    anchors = torch.cat(multi_level_anchors)
    valid_flags = torch.cat(multi_level_valid_flags)

    #得到完全在图像内部的anchor标志
    inside_flags = anchor_inside_flags(torch.tensor(anchors), torch.tensor(valid_flags),
                                           (image_heigt,image_width),)
    num_inside_flags = len(inside_flags)
    #得到完全在图像内部的anchor
    anchors_valid = anchors[inside_flags, :]
    num_anchors_valid = len(anchors_valid)
    
    '''
    观察anchors的数量变化
    >>> print(num_multi_level_anchors)
    >>> print(num_multi_level_valid_flags)
    >>> print(num_inside_flags)
    >>> print(num_anchors_valid)
    '''

    random_anchors_index = random_choice(num_choices,num_gallery=num_anchors_valid)
    random_anchors = anchors_valid[random_anchors_index]

    if isinstance(random_anchors,torch.Tensor):
        random_anchors = random_anchors.cpu().numpy()
    show_bboxes(random_anchors,imgDir,bbox_color=colors)




def main():
   #以下均为默认参数
    #要把锚框绘制到哪张图片上
    imgDir = 'demo/test/00.jpg'

    #填写该图片的尺寸信息
    image_heigt=1920
    image_width=1080

    #设置锚框的要生成锚框的基本信息
    scales=[100],#锚框面积
    ratios=[0.5,1,2],#锚框比例
    strides=[4, 8, 16, 32]#不同特征图上锚框分布的间距,要和特征图的个数相同

    #设置输入rpn中特征图的大小，一个元组代表一个特征图
    featmap_sizes = [(270, 480),(135, 24),(68, 120),(34, 60)]
    #显示多少个锚框
    num_choices=10
    
    gene_anchors(imgDir,
                 image_heigt,
                 image_width,
                 scales,
                 ratios,
                 strides,
                 num_choices,
                 featmap_sizes)

if __name__ == '__main__':
    main()

