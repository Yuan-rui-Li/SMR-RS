import cv2
import numpy as np
from PIL import Image
from functools import partial

from .image import mask_blend

#在边框的四个角上画园
def corner_circle(bbox:np.ndarray, 
                 img:np.ndarray, 
                 radius:int=5, 
                 color:tuple=(0,0.0), 
                 thickness:int=-1):

    tl=bbox[ :2].astype(np.int32)
    tr=np.array([bbox[2], bbox[1]]).astype(np.int32)
    bl=np.array([bbox[0], bbox[3]]).astype(np.int32)
    br=bbox[2:4].astype(np.int32)

    corners = [tl, tr, bl, br]
    for corner in corners:
        cv2.circle(img,tuple(corner),radius,color,thickness=thickness)
        

#产生随机颜色
def gene_color(mode:str=None, color:tuple=None):

    if(mode == None) and (color is not None):
        return color

    if(mode == 'rand') or (color == None):
	    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))


def put_Text(image:np.ndarray=None,
            font=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
            text_color=(255,128,0),
            thickness=2,
            text=None,
            coordinate=None,
            background_color=(0,47,167)):

    if text is None:
        return

    #文字的左下角坐标
    text_coordinate = coordinate
    #得到文字的尺寸大小，基线高度
    retval, baseLine = cv2.getTextSize(text,fontFace=font,fontScale=fontScale, thickness=thickness)

    tl = [text_coordinate[0],text_coordinate[1]-retval[1]]
    br = [tl[0]+retval[0],tl[1]+retval[1]+baseLine]
    cv2.rectangle(image, tuple(tl), tuple(br), color=background_color, thickness=-1)

    cv2.putText(image, 
                f'{text}',#文字内容
                coordinate,#文字放置位置坐标
                font,#字体
                fontScale=fontScale,#文字大小
                color=text_color,#文字颜色
                thickness=thickness)#文字线宽



def attach_bbox(image:np.ndarray,
                bbox:np.ndarray,#边框坐标，形如(tl_x, tl_y, br_x, br_y)
                score:float=None,#边框得分
                i:int=None,#bbox编号
                bbox_line_width:int=6,#边框线宽
                bbox_color:tuple=None,
                text_color:tuple=None,
                custom_text:str=None,
                show_score:bool=True,
                show_ratio:bool=True,
                show_index:bool=True,
                show_coordinate:bool=True,
                show_W_H:bool=True,
                show_area:bool=True,
                show_text:bool=True,
                show_center:bool=True,
                show_corners:bool=True,
                alpha_rect:bool=True):
    '''
    @brief:在给定的一张图片上画上给定的方框。
    @args:
        image(np.ndarray):可以是从cv2.imread()加载的图片
        bbox(np.narray):bbox坐标,形如(tl_x, tl_y, br_x, br_y)
    Example:
    >>> import cv2
    >>> from PIL import Image
    >>> image = cv2.imread(img_path)
    >>> attach_bbox(image, bbox, pred_score, i)
    >>> img = Image.fromarray(image)
    >>> img.show()
    '''
    #获得bbox的左上角和右下角的坐标
    tl=bbox[ :2].astype(np.int32)
    br=bbox[2:4].astype(np.int32)
    #计算边框的面积
    area = np.around((br[1]-tl[1])*(br[0]-tl[0]),2)
    #边框颜色
    if bbox_color==None:bbox_color = gene_color(color=(0,255,0))
    #绘制边框
    if alpha_rect:
        rect_image =np.zeros_like(image, dtype=np.uint8)
        cv2.rectangle(rect_image, tuple(tl), tuple(br), bbox_color, bbox_line_width)
        mask = np.sum(rect_image, -1)
        image = mask_blend(image, mask, alpha=1, color=bbox_color)
    else:
        cv2.rectangle(image, tuple(tl), tuple(br), bbox_color, bbox_line_width)
    
    #绘制边框中心
    if show_center:    
        #圆心
        center = (int((bbox[2]+bbox[0])/2), int((bbox[3]+bbox[1])/2))
        #半径
        # radius = int(np.clip((area//100000),10,1000))
        radius = 10
        #颜色
        # if area > 1920*1080 : color=(255, 100, 100)
        # else: color = (0, 0, 255)
        color = (255, 255, 255)
        cv2.circle(image,center,radius,color,thickness=-1)

    #为边框四个角画圆
    if show_corners:
        corner_circle(bbox,image,color=gene_color(color=(255,69,0)),thickness=-1,radius=10)

    #绘制文字
    if show_text:
        put_text = partial(put_Text,image)

        base_offset = 15
        text='info:'
        #文字编号
        index = 1
        put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))
        if(custom_text is not None):
            text=custom_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

        if (i is not None) and show_index:
            index_text='i:'+str(i)
            text=index_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

        if show_coordinate:
            #左上角坐标展示
            tl_text_base = 'tl:'
            text = tl_text_base+str(tl)
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

            #右下角坐标展示
            br_text_base = 'br:'
            text = br_text_base+str(br)
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

        if (score is not None) and show_score:
            score = np.around(score.astype(np.float64),2)
            score_text_base ='score:'
            score_text = score_text_base+str(score)
            text=score_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

        if show_ratio:
            #计算bbox的高宽比
            ratio = np.around((br[1]-tl[1])/(br[0]-tl[0]),2)
            ratio_text_base='ratio:'
            ratio_text=ratio_text_base+str(ratio)
            text=ratio_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

        if show_W_H:
            #计算bbox的宽
            width = br[0]-tl[0]
            width_text_base='width:'
            width_text=width_text_base+str(width)
            text=width_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

            #计算bbox的高
            height = br[1]-tl[1]
            height_text_base='height:'
            height_text=height_text_base+str(height)
            text=height_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))

        if show_area:
            area_text_base='area:'
            area_text=area_text_base+str(area)
            text=area_text
            index+=1
            #绘制文字
            put_text(text=text,coordinate=(tl[0]+5,tl[1]+base_offset*index))    