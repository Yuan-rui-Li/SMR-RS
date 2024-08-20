import cv2
import os
from typing import Union
import numpy as np
from PIL import Image
import copy

def batch_image_resize(src_path:str, dst_path:str, out_size:Union[tuple,list], inter=cv2.INTER_LINEAR):
    '''
    @brief: Resizes a batch of image. 
    @para:
        src_path(str): Input image folder path.
        dst_path(str): Output image folder path. 
        out_size: (width,height)
        inter: interpolation method, default is cv2.INTER_LINEAR
    '''
    for file_name in os.listdir(src_path):

        img_path=os.path.join(src_path,file_name)
        img_src=cv2.imread(img_path)

        img_resize = cv2.resize(img_src, out_size, interpolation=inter)

        save_path = os.path.join(dst_path,file_name)
        cv2.imwrite(save_path, img_resize)



def mask_blend(img:np.ndarray, mask:np.ndarray, alpha: float=0.6, color:tuple=([190, 255, 199])):
    '''
    @brief: implement alpha blending between mask and image
    @para:
        img:image has shape (H,W,C)
        mask:mask ha shape (H,W,C) or (H,W)
        alpha:the pixel value of blend is determine by alpha, pixel value is equal to img*(1-alpha) + mask*alpha 
        color:the mask color when blending
    @returm Image type image
    ''' 
    mask_copy = mask.copy()  
    if len(mask.shape) == 2: cover = mask_copy.copy().astype(bool) 
    else : cover = mask_copy.copy()[:,:,0].astype(bool) 
    assert img.shape[0:2] == mask.shape[0:2], 'image size and mask size must same'
    #掩膜在图片上的颜色
    color=np.array(color)
    img[cover] = (img[cover]*(1-alpha))+(color*alpha)

    output=img

    return output


