import cv2
import numpy as np
import copy

from .draw_bbox import attach_bbox


def EVENT(event, x, y,flags, param):
    #左键按下回调操作
    if (event == cv2.EVENT_LBUTTONDOWN) and (param['stop_flag']=='EVENT_LBUTTONDOWN'):
        param['coordinate']['tl'].append(x)
        param['coordinate']['tl'].append(y)
        param['stop_flag']=False

    #鼠标移动回调操作
    if (event == cv2.EVENT_MOUSEMOVE) and (param['stop_flag']==False):
        #得到左上角坐标
        bbox_tl = param['coordinate']['tl']
        #得到右下角坐标
        bbox_br = [x+1,y+1]
        #得到完整坐标
        bbox = np.array(bbox_tl+bbox_br)
        #在给定图片上显示bbox
        img =  copy.deepcopy(param['image'])
        attach_bbox(img,bbox,alpha_rect=False)
        cv2.imshow("image",img)
    #左键松开回调操作    
    if (event == cv2.EVENT_LBUTTONUP) and (param['stop_flag']==False):
        param['coordinate']['br'].append(x)
        param['coordinate']['br'].append(y)
        param['stop_flag']=True
        

def get_bbox_coordinate(image:np.ndarray):
    '''
    @brief:通过鼠标在给定的图片上选取一个方框
    @note:按下鼠标左键拖动绘制方框,按下Esc结束
    @return:
        bbox_coordinate(np.ndarray):[tl_x, tl_y, br_x, br_y]
    '''

    coordinate = dict(tl=[],br=[])
    stop_flag = 'EVENT_LBUTTONDOWN'
    param = dict(image=image, coordinate=coordinate, stop_flag=stop_flag)
    cv2.namedWindow("image",0)
    cv2.imshow("image",image)
    cv2.setMouseCallback("image", EVENT, param)
    
    while(True):
        #按下Esc退出选框
        if (cv2.waitKey(50) == 27):#//非常重要
            break
    cv2.destroyWindow('image')
 
    return np.array(param['coordinate']['tl']+param['coordinate']['br'])




