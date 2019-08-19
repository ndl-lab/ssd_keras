import numpy as np
import os
import glob
import pandas as pd
import cv2
from xml.etree import ElementTree
import random
import shutil
import glob
from PIL import Image

for dirname in os.listdir("tmp"):
    print(dirname)
    for fpath in glob.glob(os.path.join("fullsizeimg",dirname,"original","*")):
        img=Image.open(fpath)
        img_resize = img.resize((300, 300),Image.LANCZOS)
        img_resize.save(os.path.join("300_300img",dirname+"_"+os.path.basename(fpath)))