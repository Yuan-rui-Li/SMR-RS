import torch
from PIL import Image
import numpy as np
import cv2
import os
from typing import Union
from collections import Counter

from .bbox import bbox_overlaps, bbox2mask
from .file import load_eval,save_eval


def count_number(numbers:Union[list,np.ndarray], out_num:int):
    '''
    统计一组数字中相同数字出现的次数
    '''
    return Counter(numbers).most_common(out_num)


def mask2iou(mask1:np.ndarray,mask2:np.ndarray):
    """
    @bref:计算mask1和mask2的iou(intersection over union)
    @arg:
        mask1(np.ndarry[H,W])
        mask1(np.ndarry[H,W])
    @return:
        iou(float):Intersection Over Union
    """
    #得到并集
    union=mask1|mask2
    union_bool=union.astype(bool)
    #求并集区域的面积
    union_area=union_bool.sum()

    #得到交集
    inter=mask1 & mask2
    inter_bool=inter.astype(bool)
    #求相交区域的面积
    inter_area=inter_bool.sum()

    #得到iou
    iou=np.float16((inter_area+1)/(union_area+1))

    return iou



def mask2PA2CPA(mask1:np.ndarray,mask2:np.ndarray):
    """
    @bref:计算mask1和mask2的像素准确率PA(Pixel Accuracy)和类别准确率(Class Pixel Precision)
    @arg:
        mask1(np.ndarry[H,W]):标签mask
        mask1(np.ndarry[H,W]):预测得到的mask
    @return:
        pa(float):Pixel Accuracy
        cpa(float):Class Pixel Accuracy
    """

    mask1=mask1.astype(bool)
    mask2=mask2.astype(bool)

    lable_True = mask1
    lable_False = ~mask1
    infer_True = mask2
    infer_False = ~mask2

    #得到交集中的像素个数
    union=(mask1|mask2).sum()
    #所有真正像素的个数
    tp = (infer_True & lable_True).sum()
    #所有真反像素的个数
    tn = (infer_False & lable_False).sum()
    #所有假正像素的个数
    fp = (infer_True & lable_False).sum()
    #所有假反像素的个数
    fn = (infer_False & lable_True).sum()

    assert union==(tp+fp+fn),"error"

    #计算像素准确率
    pa = np.float16((tp+tn+1)/(tp+tn+fp+fn+1))
    #计算类别像素准确率
    cpa = np.float16((tp+1)/(tp+fp+1))#或者tn/(tn+fn)
    #计算召回率
    recall = np.float16((tp+1)/(tp+fn+1))

    return pa, cpa, recall


def bboxes2bboxes2iou(label_bboxes:np.ndarray, infer_bboxes:np.ndarray):
    '''
    计算一组预测框与一组标签框的IoU和Precision,原理是给标签框分配预测框
    '''
    device = torch.device('cuda:0')
    label_bboxes = torch.tensor(label_bboxes, device=device)
    infer_bboxes = torch.tensor(infer_bboxes, device=device)

    num_label_bboxes = label_bboxes.size()[0]
    num_infer_bboxes = infer_bboxes.size()[0]


    IoU, Precision = bbox_overlaps(label_bboxes[None,:,:], infer_bboxes[None,:,:-1])#None升维
    #降维
    IoU = IoU.view(num_label_bboxes, num_infer_bboxes)
    Precision = Precision.view(num_label_bboxes, num_infer_bboxes)

    if num_label_bboxes >= num_infer_bboxes:
        #得到每个infer_bbox与所有label_bbox最大的iou值，和label_bbox的索引
        bbox_ious, infer_bboxes_index = torch.max(IoU, dim=0)
        bbox_Precisions = Precision[list(infer_bboxes_index),list(range(num_infer_bboxes))]
        assert num_infer_bboxes == bbox_ious.size()[0] == bbox_Precisions.size()[0]

    else:
        #得到每个label_bbox与所有infer_bbox最大的iou值，和infer_bbox的索引
        bbox_ious, infer_bboxes_index = torch.max(IoU, dim=1)
        bbox_Precisions = Precision[list(range(num_label_bboxes)),list(infer_bboxes_index)]
        assert num_label_bboxes == bbox_ious.size()[0] == bbox_Precisions.size()[0]

    iou_result = bbox_ious.mean().cpu()
    Precision_result = bbox_Precisions.mean().cpu()

    return iou_result, Precision_result


def bbox_cal(path_1:str, path_2:str, singal_bbox2singal_bbox=False):
    '''
    @brief: calculate the iou between two batch bboxes
    @para:
        path_1: the path of a batch bboxes, txt file which include 
            a dict like {'0001.jpg':{tl_x, tl_y, br_x, br_y}}.
        path_2: reference as path_1
    '''

    label_bboxes_dict=load_eval(path_1)
    infer_bboxes_dict=load_eval(path_2)
    
    ious = []
    Precisions = []
    for file_base_name in label_bboxes_dict:
        label_bboxes = np.array(label_bboxes_dict[file_base_name])
        infer_bboxes = np.array(infer_bboxes_dict[file_base_name])
        num_label_bboxes = label_bboxes.shape[0]
        num_infer_bboxes = infer_bboxes.shape[0]

        if num_infer_bboxes == 0:
            ious.append(0)
            Precisions.append(0)
            continue

        infer_mask = np.zeros((1920, 1080),dtype=np.uint8)
        for box in infer_bboxes:
            infer_mask = bbox2mask(box, image=infer_mask)

        label_mask = np.zeros((1920, 1080),dtype=np.uint8)
        for box in label_bboxes:
            label_mask = bbox2mask(box, image=label_mask)
        
        iou = mask2iou(infer_mask, label_mask)
    
        ious.append(iou)

    ave_iou = np.float16(sum(ious)/len(ious))
    # ave_Precision = np.float16(sum(Precisions)/len(Precisions))

    # print('******bbox_IoU part******')
    # print(f'iou--{(ave_iou)}\n')

    # print('******bbox_Precision part******')
    # print(f'Precision--{(ave_Precision)}\n')

    return ave_iou

    #计算模型评价指标
def mask_cal(path_1:str, path_2:str):
    '''
    @brief:计算两个文件夹内对应mask图片的吻合指标
    '''

    #标签mask存放列表
    label_mask_list=[]
    #标签mask文件名存放列表
    label_name_list=[]
    #推理结果mask存放列表
    infer_mask_list=[]
    #推理结果文件名存放列表
    infer_name_list=[]
    for file_name in os.listdir(path_1):
        #读取标签mask
        with Image.open(os.path.join(path_1,file_name)) as label_mask:
            label_mask = np.asarray(label_mask)
        label_mask_list.append(label_mask)   
        label_name_list.append(file_name)
            
        #读取推理结果mask
        with Image.open(os.path.join(path_2,file_name)) as infer_mask:
            infer_mask = np.asarray(infer_mask)
        infer_mask_list.append(infer_mask)
        infer_name_list.append(file_name)

    assert len(label_mask_list)==len(infer_mask_list),"错误:两个文件夹内的文件个数不相等"

    # pa, cpa, recall = mask2PA2CPA(label_mask_list[0],infer_mask_list[0])
    # print(infer_name_list[0])
    # print(pa)
    # print(cpa)

    IoUs=[]
    PAs=[]
    CPAs=[]
    recalls = []
    num_masks=len(label_mask_list)
    for i in range(0,num_masks,1):
        iou=mask2iou(label_mask_list[i],infer_mask_list[i])
        pa, cpa, recall = mask2PA2CPA(label_mask_list[i],infer_mask_list[i])
        IoUs.append(iou)
        print(f'{infer_name_list[i]}-mask-IoU:{IoUs[i]}\n')
        PAs.append(pa)
        print(f'{infer_name_list[i]}-mak-PA:{PAs[i]}\n')
        CPAs.append(cpa)
        print(f'{infer_name_list[i]}-mask-CPA:{CPAs[i]}\n')
        recalls.append(recall)
    
    MIoU =np.float16(sum(IoUs)/num_masks)
    # print('******Mask-IoU part******')
    # print(f'MIoU:{MIoU}\n')

    # print('\n******Mask-PA part******')
    mpa = np.float16(sum(PAs)/num_masks)
    # print(f'MPA:{mpa}\n')

    # print('\n******Mask-Precision part******')
    mcpa = np.float16(sum(CPAs)/num_masks)
    # print(f'MCPA:{mcpa}\n')

    # print('\n******Mask-recall part******')
    mrecall = np.float16(sum(recalls)/num_masks)
    # print(f'MRecall:{mrecall}\n')

    return MIoU, mpa, mcpa, mrecall





