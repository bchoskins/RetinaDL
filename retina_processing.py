# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
import pathlib


#files = filenames[0:9]
#files = glob.glob('/Dedicated/jmichaelson-sdata/UK_Biobank/image_retina/21016/4852586_21016_1_0.png')
#for file in filenames:
    #image = plt.imread(file)
    #plt.imshow(image)
    #path = pathlib.PurePath(file)
    #plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/images/" + path.name)
    #print("image", file,"loaded")

#files2 = glob.glob("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/images/*.png")

filenames = glob.glob('/Dedicated/jmichaelson-sdata/UK_Biobank/image_retina/21016/*.png')

def rescaler_128(x):
    np_image= np.array(x)
    image_size = np_image.shape
    f1 = 128/image_size[0] ## X direction
    f2 = 128/image_size[1] ## Y direction
    f3 = 1
    np_image = ndimage.zoom(np_image,(f1,f2,f3))
    #print("rescaled!")
    return np_image

def process_images(filenames):
    labels = []
    images = list()
    os.chdir("/Dedicated/jmichaelson-sdata/UK_Biobank/image_retina/21016")
    i=0
    for file in filenames:
        if file.endswith(".png"):
            image = plt.imread(file)
            rescaled_image_128 = rescaler_128(image)
            images.append(rescaled_image_128)
            label = file.split('/')[-1].split('_')[0]  
            labels.append(label)
            i=i+1
            print(i)
    images = np.array(images)
    labels = np.array(labels)
    np.save("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/retina_images_"+"full"+".npy", images)
    np.save("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/labels_"+"full"+".npy",labels)
    msg = 'images/labels loaded'
    return msg
    
process_images(filenames)