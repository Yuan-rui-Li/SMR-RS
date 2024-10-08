import numpy as np
from typing import Union
import torch
import cv2
from PIL import Image

from .file import load_eval

def bbox2mask(bbox:np.array, mask_size:Union[list,tuple]=None, image:np.array=None):
    '''
    @brief:Convert a bbox area to mask on a zero binary-image which has specified size.
    @args:
        bbox: Contains coordinates of one bbox, it shold has the format: array([tl_x, tl_y, br_x, br_y]).
        mask_size: like(height, width), Indicates the size of the zero binary-image, when image_size or image is None, the other must be not None.  
        image: Image which mask generate on it, when image_size or image is None, the other must be not None.  
    @return:
        img(np.ndarray):A binary-image,it's bbox area has been replaced to 255.
    '''

    if mask_size is not None:
        assert image == None, 'when image_size is not None, image must be None.'
        img = np.zeros(mask_size,dtype=np.uint8)
    else:
        assert image is not None, 'when image_size is None, image must not be None.'
        img = image

    tl = bbox[0:2]
    br = bbox[2:]
    
    img[tl[1]:br[1],tl[0]:br[0]] = 255

    return img


def bboxes2mask(src_path:str,save_path:str):
    '''
    @brief: Convert a bbox area to mask on zero-image and save at a specified path.
    @para:
        src_path: source path of the bboxes which is a txt file
                  include a dict like {'0001.jpg':[[10,10,20,20],],}
        save_path: the saving path of results which is a batch 
                   binary-image covered with bbox mask
    '''
    #推理mask保存路径
    # bboxes_save_path = path['bboxes_save_path']
    data_dict=load_eval(src_path)
    for file_base_name in data_dict:
        bboxes = np.array(data_dict[file_base_name])
        num_bboxes = bboxes.shape[0]
        img = np.zeros((1920,1080),dtype=np.uint8)
        for i in range(num_bboxes):
            bbox = bboxes[i]
            bbox2mask(bbox,image=img)
        img=Image.fromarray(img)
        print(f'one image has been saved to{save_path+file_base_name}.\n')
        img.save(save_path+file_base_name)

# Copyright (c) OpenMMLab. All rights reserved.



def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        """
        torch.clamp使用示例:
            torch.clamp(input, min, max, out=None) → Tensor
            将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。

        操作定义如下：
                    | min, if x_i < min
                y_i=| x_i, if min <= x_i <= max
                    | max, if x_i > max
        """
        #要先将tensor转换成float()类型，再进行clamp操作
        return x.float().clamp(min, max).half()#tensor.half():将原始张量转换为半精度类型

    return x.clamp(min, max)

class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]#...代表省略前面的所有维
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)#返回tensor[ious] 形状为size[len(bboxes1),len(bboxes2)]

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    #假设坐标原点在左上角，前两列为左上角坐标，后两列为右下角坐标，形如[tl_x,tl_y,bl_x,bl_y]
    #area1和area2相较与bboxes1和bboxes2少一维
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (#计算gt_bboxes面积，tensor的size()不变
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (#计算bboxes面积
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        #得到相较于gt_bboxes中的每个anchors，在bboxes中所有的anchor与之重合部分的坐标
        #lt：若重合，则是重合部分的左上角坐标。rb：若重合，则是重合部分的右下角坐标
        lt = torch.max(bboxes1[..., :, None, :2],#...,可以忽视   广播机制，广播成-高度为bboxes1中anchor的个数，行数为bboxes2中anchor的个数
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        #rb - lt会将非与gt_bboxes重合的bboxes的高或宽变为0
        wh = fp16_clamp(rb - lt, min=0)
        #得到相较于gt_bboxes中的每个anchors，在bboxes中所有的anchor与之重合部分的面积
        overlap = wh[..., 0] * wh[..., 1]#宽*高=面积,降维，overlap的size=[bboxes1中anchor的个数，bboxes1中anchor的个数]

        #area1和area2的size为一维
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)#将最小面积控制在eps

    #overlap中的0值代表不与gt_bboxes的bboxes，0除任何值为0,故将此处的union值置0,代表不与gt_bboxes相交
    ious = overlap / union
    if mode in ['iou', 'iof']:
        #返回iou和precision
        return ious, overlap / area2 
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def bbox2pad(bboxes:np.ndarray, img:np.ndarray, empty_part:int=30, pad_color:tuple=(0,0,0)):
    '''
    @breif pad image to make it could contain the full bboxes
    @para:
        bboxes: has the shape of (N, 4), N indicates the number of bboxes,
            4 contains the coordinates of bboxes like (tl_x, tl_y, br_x, br_y)
        img: the image which bboxes will put on it.
        empty_part: the extra part when padding to contral the distance from the bboxes to border.
        pad_color: the color of padiing part.
    '''
    h, w, c = img.shape
    #上方超出的距离
    top = bboxes[:,1].min()
    if top < 0: top_pad = -top
    else: top_pad = 0
    top_pad+=empty_part
    
    #下方超出的记录
    bottom = bboxes[:,3].max()
    if (bottom) > h: bottom_pad = bottom-h
    else: bottom_pad = 0
    bottom_pad+=empty_part
    
    #左边超出的距离
    left = bboxes[:,0].min()
    if left < 0: left_pad = -left
    else: left_pad = 0
    left_pad+=empty_part
    
    #右边超出的距离
    right = bboxes[:,2].max()
    if (right) > w: right_pad = right-w
    else: right_pad = 0
    right_pad+=empty_part

    img = cv2.copyMakeBorder(img.copy(), int(top_pad), int(bottom_pad), int(left_pad), int(right_pad), cv2.BORDER_CONSTANT, value=pad_color)

    #对bboxes进行偏移，以保证坐标原点不变
    bboxes[:,1]+=top_pad
    bboxes[:,3]+=top_pad
    bboxes[:,0]+=left_pad
    bboxes[:,2]+=left_pad
    return img, bboxes

