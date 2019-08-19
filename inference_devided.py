# -*- coding:utf-8 -*-
import cv2
import keras
import os
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
import gc
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf
import glob
import sys
import json
from keras import backend as K
K.clear_session()

from ssd_tools.ssd import SSD300
from ssd_tools.ssd_training import MultiboxLoss
from ssd_tools.ssd_utils import BBoxUtility
from PIL import Image



np.set_printoptions(suppress=True)

# パラメータ
batch_size=10
NUM_CLASSES = 2
input_shape = (300, 300, 3)


model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(os.path.join('ssd_tools','weights.hdf5'), by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)


dpiinfo={}


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def main():
    imgpathlist=list(glob.glob(os.path.join(path_test_prefix,"*")))
    cnt = 0
    while cnt<len(imgpathlist):
        inputs = []
        images = []
        filenames = []
        for img_path in imgpathlist[cnt:min(cnt+batch_size,len(imgpathlist))]:
            print(img_path)
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            images.append(imread(img_path,mode="RGB"))
            inputs.append(img.copy())
            filenames.append(os.path.basename(img_path))
        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=1, verbose=1)
        results = bbox_util.detection_out(preds)
        cnt += batch_size
        for i, img in enumerate(images):
            print(i)
            # Parse the outputs.
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_xmax = results[i][:, 4]

            # Get detections with confidence higher than 0.2.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]
            if len(top_indices)==0:
                cvimg = np.asarray(img)
                im = cv2pil(cvimg)
                im.save(os.path.join("output", filenames[i] + "_00.jpg"),
                         dpi=(dpiinfo["width_dpi"], dpiinfo["height_dpi"]), quality=100)
                continue
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_xmax = det_xmax[top_indices]
            div_x=0
            for j in range(top_conf.shape[0]):
                print(top_conf[j])
                if j>=1:
                    break
                xmin = int(round(top_xmin[j] * img.shape[1]))
                xmax = int(round(top_xmax[j] * img.shape[1]))
                div_x=(xmin+xmax)//2
            cvimg = np.asarray(img)
            im1=cv2pil(cvimg[:,:div_x,::-1])
            im2=cv2pil(cvimg[:, div_x:, ::-1])
            im1.save(os.path.join("inference_output",filenames[i]+"_01.jpg"), dpi=(dpiinfo["width_dpi"],dpiinfo["height_dpi"]),quality=100)
            im2.save(os.path.join("inference_output", filenames[i] + "_02.jpg"), dpi=(dpiinfo["width_dpi"],dpiinfo["height_dpi"]),quality=100)

        del inputs,images
        gc.collect()



if __name__ == '__main__':
    with open(os.path.join('ssd_tools','dpiconfig.json'))as f:
        dpiinfo = json.load(f)
    path_test_prefix="inference_input"
    main()

