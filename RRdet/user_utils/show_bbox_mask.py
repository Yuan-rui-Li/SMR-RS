import cv2
import numpy as np
from PIL import Image
import copy
import torch
from typing import Tuple
import os

from .draw_bbox import attach_bbox, put_Text
from .image import mask_blend
from .file import new_path


def show_bboxes(bboxes:np.ndarray,
                image:Tuple[np.ndarray, str]=None,
                merge_bboxes:bool=True,
                text:str=None,
                show_text:str=True,
                bbox_color:Tuple[tuple, list]=(255, 0, 0),
                bbox_line_width:int=3,
                save_path:str=None,
                show_coners:bool=True):
    if save_path is not None:
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0],exist_ok=True)
    else:
        save_path = 'anchors_show.jpg'
        
    if type(bbox_color) is tuple: bbox_color=[bbox_color]

    if isinstance(image,str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    num_bboxes = len(bboxes)
    if num_bboxes > len(bbox_color) and bbox_color is not None: 
        print("Colors is not enough, please add some colors.")
        return
    if len(bboxes.shape) == 1:
        bboxes = np.array([bboxes,])
        num_bboxes = 1

    if merge_bboxes:
        img = copy.deepcopy(image)
        for i, bbox in enumerate(bboxes):
            attach_bbox(img,
                        bbox,
                        i=i,
                        custom_text=text, 
                        show_text=show_text, 
                        bbox_line_width=bbox_line_width,
                        bbox_color=bbox_color[i],
                        show_corners=show_coners)
        img = Image.fromarray(img)
        img.show()
        if save_path is not None:
            img.save(save_path)
            print(f'Result saved at {save_path}.')
    
    else:
        for i, bbox in enumerate(bboxes):
            img = copy.deepcopy(image)
            attach_bbox(img,
                        bbox,
                        i=i,
                        custom_text=text, 
                        show_text=show_text, 
                        bbox_line_width=bbox_line_width,
                        bbox_color=bbox_color[i],
                        show_corners=show_coners)
            img = Image.fromarray(img)
            img.show()
            if save_path is not None:
                new_path = 'i_'+save_path
                img.save(new_path)
                print(f'Result saved at {new_path}.')



def show_mask_singal(mask:np.ndarray, 
                     image:np.ndarray=None, 
                     mask_index:str=None, 
                     text:str=None, 
                     save_path:str=None):

    mask=np.array(mask).astype(np.uint8)*255
    put_Text(image=mask,text=text,coordinate=(0,120),fontScale=5,thickness=3)
    img = Image.fromarray(mask)
    img.show()
    if save_path is not None:
            img.save(save_path)


def show_bbox_singal(image:np.ndarray=None, 
                     bbox:np.ndarray=None, 
                     bbox_score:float=0, 
                     i:int=None, 
                     show_text:bool=True, 
                     save_path:str=None):

    img = copy.deepcopy(image)
    attach_bbox(img ,bbox,bbox_score,i=i, show_text=show_text)
    img_PIL = Image.fromarray(img)
    img_PIL.show()
    if save_path is not None:
            img_PIL.save(save_path)


def show_masks_bboxes(img:np.ndarray,
                     masks:list, 
                     bboxes:torch.tensor,
                     bbox_scores:torch.tensor,
                     num_masks:int, 
                     num_bboxes:int, 
                     merge_masks:bool, 
                     merge_bboxes:bool,
                     bbox_color:Tuple[tuple, list]=(255, 0, 0),
                     bbox_line_width:int=6, 
                     show_bboxes:bool=True,
                     show_masks:bool=True,
                     show_blend_result:bool=True,
                     blend_bboxes_masks:bool=True,
                     show_text:bool=True,
                     bbox_save_path:str=None,
                     mask_save_path:str=None,
                     blend_save_path:str=None,):

    if type(bbox_color) is tuple: bbox_color=[bbox_color]

    num_bboxes = len(bboxes)
    if num_bboxes > 3: bbox_color=[bbox_color[0] for i in range(num_bboxes)]

    img_blend = img.copy()
    #show masks
    if show_masks:
        if (merge_masks) and (num_masks>1):
            # t1 = time.time()
            masks_sum=copy.deepcopy(masks[0])
            for i in range(num_masks-1):
                masks_sum|=masks[i+1]
            masks = [masks_sum]
            num_masks = 1
            # t2 = time.time()
            # t = t2-t1
            # print(t)

        if (num_masks==1):
            mask = masks[0].cpu().numpy()
            show_mask_singal(mask, save_path=mask_save_path)
        else:
            for i in range(num_masks):
                mask = masks[i].cpu().numpy()
                #获取文件保存路径
                mask_path = mask_save_path
                if mask_save_path is not None:
                    mask_path = new_path(mask_save_path, i)
                show_mask_singal(mask,text=str(i), save_path=mask_path)
    
    #show bboxes
    if show_bboxes:
        bboxes = bboxes.cpu().numpy()
        bbox_scores = bbox_scores.cpu().numpy()
        if (not merge_bboxes) and (num_bboxes>1):
            for i in range(num_bboxes):
                bbox = bboxes[i]
                bbox_score = bbox_scores[i]
                #获取文件保存路径
                bbox_path = bbox_save_path
                if bbox_save_path is not None:
                    bbox_path = new_path(bbox_save_path, i)
                # if num_bboxes>3: bbox_color.append(bbox_color[0])
                show_bbox_singal(image=img,
                                 bbox=bbox,
                                 bbox_color=bbox_color[i],
                                 bbox_score=bbox_score,
                                 i=i,
                                 bbox_line_width=bbox_line_width,
                                 save_path=bbox_path)

        else:
            for i in range(num_bboxes):
                bbox = bboxes[i]
                bbox_score = bbox_scores[i]
                # if num_bboxes>3: bbox_color.append(bbox_color[0])
                attach_bbox(img,
                            bbox,
                            bbox_score,
                            i=i,
                            bbox_color=bbox_color[i],
                            show_text=show_text,
                            bbox_line_width=bbox_line_width)
            img_PIL = Image.fromarray(img)
            img_PIL.show()
            if bbox_save_path is not None:
                    img_PIL.save(bbox_save_path)

    #blend masks with image
    if show_blend_result:
        alpha = 0.5
        masks=np.array(masks[0].cpu()).astype(np.uint8)*255
        if blend_bboxes_masks: blend_result = mask_blend(img, masks, alpha=alpha)
        else: blend_result = mask_blend(img_blend , masks, alpha=alpha)
        blend_result = Image.fromarray(blend_result)
        blend_result.show()
        if blend_save_path is not None:
            blend_result.save(blend_save_path)