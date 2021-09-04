# 基于mediapipe的人像语义分割
import cv2
import mediapipe as mp
import numpy as np
import os
import random
from argparse import ArgumentParser

import parse as parse
from pandas.tests.io.test_parquet import pa

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def show(img):
    cv2.imshow('', img)
    cv2.waitKey()


# 输入静态图像，纯色填充标注识别出的前景背景
def staticSegmentation(
        inputdirpath='./inputimage',  # 输入图片文件夹，输入图片中不要含有中文名
        outputdirpath='./outputimage',# 输出图片文件夹
        bgcolor=(100, 150, 200),      # 输出图片背景填充色
        maskcolor=(255, 255, 255)     # 输出图片掩膜填充色
):
    # 参数非法性处理
    if bgcolor is None:
        bgcolor = (192, 192, 192)
    if maskcolor is None:
        maskcolor = (255, 255, 255)

    imagefiles = [os.path.join(inputdirpath, each) for each in os.listdir(inputdirpath)]
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        for idx, file in enumerate(imagefiles):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # BGR -> RGB
            results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # Generate solid color images for showing the output selfie segmentation mask.
            fg_image = np.zeros(image.shape, dtype=np.uint8)
            fg_image[:] = maskcolor
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = bgcolor
            output_image = np.where(condition, fg_image, bg_image)
            path, filename = os.path.split(file)
            cv2.imwrite(os.path.join(outputdirpath, filename), output_image)

# 实时摄像，设置虚拟背景以及模糊效果
def dynamicSegmentation(
        # 【备注】参数解读
        # bg_mode=0，bg_para为图片路径,'*.jpg/png…'       ->设置图片背景
        # bg_mode=1，bg_para为高斯核大小,33以上奇数         ->设置虚化背景
        # bg_mode=2，bg_para为三元组(r,g,b)               ->设置纯色背景
        bg_mode=2,bg_para=(192, 192, 192),
        outputdirpath='./outputvideo/video.mp4'
):
    # 参数非法性处理
    if bg_mode is None:
        bg_mode = random.choice(range(3))
    # 随机赋予一个参数给bg_para
    if bg_mode==0:
        bgfolder = './background'
        imgfiles = os.listdir(bgfolder)
        bg_para = os.path.join(bgfolder,random.choice(imgfiles)) if len(imgfiles) else ''
    if bg_mode==1:
        bg_para = random.choice(range(31,101,2))
    if bg_mode==2:
        bg_para = (random.choice(range(256)), random.choice(range(256)), random.choice(range(256)))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video = cv2.VideoWriter(outputdirpath, fourcc, fps, size)
    cap = cv2.VideoCapture(0)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue  # If loading a video, use 'break' instead of 'continue'.
            # BGR -> RGB，且水平翻转图像
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = selfie_segmentation.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

            def check(r,g,b):
                return isinstance(r,int) and 0<=r<=255 and \
                       isinstance(g,int) and 0<=g<=255 and \
                       isinstance(b, int) and 0 <= b <= 255


            fail = False
            if bg_mode == 0:
                h, w = image.shape[:-1]
                bg_image = cv2.imread(bg_para)
                if bg_image is None:
                    fail = True
                else:
                    bg_image = cv2.resize(bg_image, (w, h))
                    bg_image = np.array(bg_image,dtype=np.uint8)
            elif bg_mode == 1:
                if bg_para % 2:
                    bg_image = cv2.GaussianBlur(image, (bg_para,bg_para), 0)
                else:
                    fail = True
            elif bg_mode == 2:
                r,g,b = bg_para
                if check(r,g,b):
                    bgcolor = (r,g,b),
                    bg_image = np.zeros(image.shape, dtype=np.uint8)
                    bg_image[:] = bgcolor
                else:
                    fail = True
            if not 0<=bg_mode<=2 or fail:
                bgcolor = (192, 192, 192),
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = bgcolor

            output_image = np.where(condition, image, bg_image)
            video.write(output_image)
            cv2.imshow('MediaPipe Selfie Segmentation', output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    video.release()
    print('Save video successfully!')
if __name__ == '__main__':
    parse = ArgumentParser(description='Semantic segmentation using mediapipe')
    parse.add_argument('--choice',default=0,type=str,help='"image" or "video"')#required=True,
    parse.add_argument('--bgcolor',type=int,help='backgroundcolor(R,G,B)',nargs = 3)
    parse.add_argument('--maskcolor',type=int,help='maskcolor(R,G,B)',nargs = 3)
    parse.add_argument('--bg_mode',type=int,help='0:image background\n1:blurred background\n2:pure background')
    # parse.add_argument('--bg_para',help='image path for image background\nor kernel size for blurred background\nor RGB tuple for pure background')
    args = parse.parse_args()
    choice = args.choice
    bgcolor = args.bgcolor
    maskcolor = args.maskcolor
    bg_mode = args.bg_mode
    if choice =="image":
        staticSegmentation(bgcolor=bgcolor,maskcolor = maskcolor)
    if choice == 'video':
        dynamicSegmentation(bg_mode = bg_mode)