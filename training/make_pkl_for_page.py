import numpy as np
import os
import glob
import pandas as pd
import cv2
from xml.etree import ElementTree
import random
import shutil
st=set()
random.seed(77)

#各画像のレイアウトの情報(今回はのど元の矩形)を、画像の左上を(0,0),右下を(1,1)となるように表した座標をpklファイルに保存する。
#pklファイルはディクショナリをダンプしたもので、キーがファイル名、バリューは矩形の座標(左上xyと右下xy)の配列とラベルの配列を持つ。

class CSV_preprocessor(object):

    def __init__(self):
        self.num_classes = 1
        self.data = dict()
        self._preprocess_CSV()
    def _preprocess_CSV(self):
        df=pd.read_table("image.tsv",names=('filename',"roll"))
        for index, row in df.iterrows():
            filename=row["filename"]
            xminp=(0.5 - row["roll"])-0.01#のど元の中心から左右に画像幅の1%ずつ広げた、短冊状の矩形を見つけたい領域とする。
            xmaxp = xminp + 0.02
            yminp=0
            ymaxp=1
            bounding_box = [xminp, yminp, xmaxp, ymaxp]
            bounding_boxes_np = np.asarray([bounding_box])
            image_data = np.hstack((bounding_boxes_np, [[1]]))
            self.data[filename] = image_data

## example on how to use it
import pickle
data = CSV_preprocessor().data
pickle.dump(data,open('page_layout.pkl','wb'))
f = open('page_layout.pkl', 'rb')


