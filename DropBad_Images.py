#Get rid of bad images
import pandas as pd 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.exposure import equalize_hist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/labels_full.npy")
####was overwritten go back to retina_processing.py
images2 = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/retina_images_full.npy")
#array([[ 0.,  0., -0., ..., -0., -0.,  0.],
       [-0.,  0.,  0., ...,  0.,  0.,  0.],
       [-0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [-0.,  0.,  0., ..., -0., -0.,  0.],
       [ 0.,  0.,  0., ..., -0., -0.,  0.],
       [ 0., -0., -0., ...,  0.,  0.,  0.]], dtype=float32)


#Get rid of bad images
    #imgsSC = []
    #labelsSC = []
    #for i in range(len(images)):
    #imgSO = images[i,:,:,0]
    #imgSO = imgSO.flatten()
    #print(type(imgSO))
    #print(imgSO.shape)
    #avg = np.average(imgSO)
    #print(avg)
    #if avg >0.2:
        #imgsSC.append(equalize_hist(images[i]))
        #print("image appended")
        #labelsSC.append(labels[i])

    #fig, axes = plt.subplots(2,10, dpi=300)
    #axes = np.hstack(axes)

    #for i in range(20):
        #axes[i].imshow(imgsSC[i])

#Figure out how to plot bad images on top and good on bottom for first 10 (put in small loop in loop above)
    goodimgsSC = []
    badimgsSC = []
    goodlabelsSC = []
    badlabelsSC = []
    for i in range(len(images)):
        imgSO = images[i,:,:,0]
        imgSO = imgSO.flatten()
        print(type(imgSO))
        print(imgSO.shape)
        avg = np.average(imgSO)
        print(avg)
        if avg >0.2:
            goodimgsSC.append(equalize_hist(images[i]))
            print("image appended")
            goodlabelsSC.append(labels[i])
        else: 
            badimgsSC.append(equalize_hist(images[i]))
            print("image appended")
            badlabelsSC.append(labels[i])

    fig, axes = plt.subplots(2,10, dpi=300)
    axes = np.hstack(axes)

    for i in range(0,10):
        axes[i].imshow(goodimgsSC[i])

    for i in range(10,20):
        axes[i].imshow(badimgsSC[i])

    np.min(goodimgsSC[1])
    np.min(badimgsSC[1])

    plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "test2.png")

#check mean of each color value
    imgR = images[:,:,:,0]
    imgG = images[:,:,:,1]
    imgB = images[:,:,:,2]

    imgR = imgR.flatten()
    imgG = imgG.flatten()
    imgB = imgB.flatten()

    avgR = np.average(imgR)
    avgG = np.average(imgG)
    avgB = np.average(imgB)

    #Check standard devaition from mean of each color value
    sd_R = np.std(imgR)
    sd_G = np.std(imgG)
    sd_B = np.std(imgB)

    print("Red Mean: % s" % (avgR) + ", Red Standard Deviation: % s"  % (sd_R))
    print("Green Mean: % s" % (avgG) + ", Green Standard Deviation: % s"  % (sd_G))
    print("Blue Mean: % s" % (avgB) + ", Blue Standard Deviation: % s"  % (sd_B))
    #Red Mean: 0.28799817, Red Standard Deviation: (+/-) 0.37072635
    #Green Mean: 0.14884506, Green Standard Deviation: (+/-) 0.19627
    #Blue Mean: 0.055384334, Blue Standard Deviation: (+/-) 0.094425365

#Specific to goodImgsSC[6] ^^^^
    img = goodimgsSC[6]

    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]

    imgR = imgR.flatten()
    imgG = imgG.flatten()
    imgB = imgB.flatten()

    avgR = np.average(imgR)
    avgG = np.average(imgG)
    avgB = np.average(imgB)

    #Check standard devaition from mean of each color value
    sd_R = np.std(imgR)
    sd_G = np.std(imgG)
    sd_B = np.std(imgB)

    print("Red Mean: % s" % (avgR) + ", Red Standard Deviation: % s"  % (sd_R))
    print("Green Mean: % s" % (avgG) + ", Green Standard Deviation: % s"  % (sd_G))
    print("Blue Mean: % s" % (avgB) + ", Blue Standard Deviation: % s"  % (sd_B))
    #Red Mean: 0.6348095028134961, Red Standard Deviation: 0.27128992818599934
    #Green Mean: 0.5629507928476494, Green Standard Deviation: 0.196936685392838
    #Blue Mean: 0.4899909573539647, Blue Standard Deviation: 0.12043989372233882

#now drop images based on distribution of RGB values of our 'good' image
    R_AVG = 0.6348095028134961
    R_SD = 0.27128992818599934
    R_L = R_AVG - R_SD
    R_R = R_AVG + R_SD
    G_AVG = 0.5629507928476494
    G_SD = 0.196936685392838
    G_L = G_AVG - G_SD
    G_R = G_AVG + G_SD
    B_AVG = 0.4899909573539647
    B_SD = 0.12043989372233882
    B_L = B_AVG - B_SD
    B_R = B_AVG + B_SD

    goodimgsSC = []
    badimgsSC = []
    goodlabelsSC = []
    badlabelsSC = []
    for i in range(len(images)):
        imageR = images[i,:,:,0]
        imageG = images[i,:,:,1]
        imageB = images[i,:,:,2]
        
        imageR = imageR.flatten()
        imageG = imageG.flatten()
        imageB = imageB.flatten()
        
        avg_R = np.average(imageR)
        avg_G = np.average(imageG)
        avg_B = np.average(imageB)
        #if (avg_R >= R_L and avg_R <= R_R) and (avg_G >= G_L and avg_G <= G_R) and (avg_B >= B_L and avg_B <= B_R):
        if (avg_R in np.arange(R_L, R_R, 0.1)) and (avg_G in np.arange(G_L,G_R,0.1)) and (avg_B in np.arange(B_L, B_R,0.1)):
            goodimgsSC.append(equalize_hist(images[i]))
            print("good image appended")
            goodlabelsSC.append(labels[i])
        else: 
            badimgsSC.append(equalize_hist(images[i]))
            print("bad image appended")
            badlabelsSC.append(labels[i])
            
    fig, axes = plt.subplots(2,10, dpi=300)
    axes = np.hstack(axes)

    for i in range(0,10):
        axes[i].imshow(goodimgsSC[i])

    for i in range(10,20):
        axes[i].imshow(badimgsSC[i])

    plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "test5.png")

#Testing above but in range of all images mean/standard deviation
    R_AVG = 0.28799817
    R_SD = 0.37072635
    R_L = R_AVG - R_SD
    R_R = R_AVG + R_SD
    G_AVG = 0.14884506
    G_SD = 0.19627
    G_L = G_AVG - G_SD
    G_R = G_AVG + G_SD
    B_AVG = 0.055384334
    B_SD = 0.094425365
    B_L = B_AVG - B_SD
    B_R = B_AVG + B_SD

    goodimgsSC = []
    badimgsSC = []
    goodlabelsSC = []
    badlabelsSC = []
    for i in range(len(images)):
        imageR = images[i,:,:,0]
        imageG = images[i,:,:,1]
        imageB = images[i,:,:,2]
        
        imageR = imageR.flatten()
        imageG = imageG.flatten()
        imageB = imageB.flatten()
        
        avg_R = np.average(imageR)
        avg_G = np.average(imageG)
        avg_B = np.average(imageB)

        if (avg_R >= R_L and avg_R <= R_R) and (avg_G >= G_L and avg_G <= G_R) and (avg_B >= B_L and avg_B <= B_R):
        #if (avg_R in np.arange(R_L, R_R, 0.01)) and (avg_G in np.arange(G_L,G_R,0.01)) and (avg_B in np.arange(B_L, B_R,0.01)):
            goodimgsSC.append(equalize_hist(images[i]))
            print("good image appended")
            goodlabelsSC.append(labels[i])
        else: 
            badimgsSC.append(equalize_hist(images[i]))
            print("bad image appended")
            badlabelsSC.append(labels[i])
            
    fig, axes = plt.subplots(2,10, dpi=300)
    axes = np.hstack(axes)

    for i in range(0,10):
        axes[i].imshow(goodimgsSC[i])

    for i in range(10,20):
        axes[i].imshow(badimgsSC[i])

    plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "test6.png")

#now test equalziing images first and then filtering from mean/standard deviation of RGB values from whole image set
    IMAGES = equalize_hist(images)

    imgR = IMAGES[:,:,:,0]
    imgG = IMAGES[:,:,:,1]
    imgB = IMAGES[:,:,:,2]

    imgR = imgR.flatten()
    imgG = imgG.flatten()
    imgB = imgB.flatten()

    avgR = np.average(imgR)
    avgG = np.average(imgG)
    avgB = np.average(imgB)

    #Check standard devaition from mean of each color value
    sd_R = np.std(imgR)
    sd_G = np.std(imgG)
    sd_B = np.std(imgB)

    print("Red Mean: % s" % (avgR) + ", Red Standard Deviation: % s"  % (sd_R))
    print("Green Mean: % s" % (avgG) + ", Green Standard Deviation: % s"  % (sd_G))
    print("Blue Mean: % s" % (avgB) + ", Blue Standard Deviation: % s"  % (sd_B))
    #Red Mean: 0.6951753786291618, Red Standard Deviation: 0.18852532486597123
    #Green Mean: 0.6497519037127439, Green Standard Deviation: 0.1378884056131113
    #Blue Mean: 0.5860665576377558, Blue Standard Deviation: 0.07728745325399138

    R_AVG = 0.6951753786291618
    R_SD =0.18852532486597123
    R_L = R_AVG - R_SD
    R_R = R_AVG + R_SD
    G_AVG = 0.649751903712743
    G_SD = 0.1378884056131113
    G_L = G_AVG - G_SD
    G_R = G_AVG + G_SD
    B_AVG = 0.5860665576377558
    B_SD = 0.07728745325399138
    B_L = B_AVG - B_SD
    B_R = B_AVG + B_SD

    goodimgsSC = []
    badimgsSC = []
    goodlabelsSC = []
    badlabelsSC = []
    for i in range(len(IMAGES)):
        imageR = IMAGES[i,:,:,0]
        imageG = IMAGES[i,:,:,1]
        imageB = IMAGES[i,:,:,2]
        
        imageR = imageR.flatten()
        imageG = imageG.flatten()
        imageB = imageB.flatten()
        
        avg_R = np.average(imageR)
        avg_G = np.average(imageG)
        avg_B = np.average(imageB)

        if (avg_R >= R_L and avg_R <= R_R) and (avg_G >= G_L and avg_G <= G_R) and (avg_B >= B_L and avg_B <= B_R):
            goodimgsSC.append(IMAGES[i])
            print("good image appended")
            goodlabelsSC.append(labels[i])
        else: 
            badimgsSC.append(IMAGES[i])
            print("bad image appended")
            badlabelsSC.append(labels[i])

    #len(goodimgsSC)
    #86974
    #len(badimgsSC)
    #1276
    plt.figure()
    fig, axes = plt.subplots(2,10, dpi=300)
    axes = np.hstack(axes)

    for i in range(0,10):
        axes[i].imshow(goodimgsSC[i])

    for i in range(10,20):
        axes[i].imshow(badimgsSC[i])

    plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "testehist8.png")

#original images with no eqaulization (mean/standard deviation filter)
    imgR = images[:,:,:,0]
    imgG = images[:,:,:,1]
    imgB = images[:,:,:,2]

    imgR = imgR.flatten()
    imgG = imgG.flatten()
    imgB = imgB.flatten()

    avgR = np.average(imgR)
    avgG = np.average(imgG)
    avgB = np.average(imgB)

    #Check standard devaition from mean of each color value
    sd_R = np.std(imgR)
    sd_G = np.std(imgG)
    sd_B = np.std(imgB)

    print("Red Mean: % s" % (avgR) + ", Red Standard Deviation: % s"  % (sd_R))
    print("Green Mean: % s" % (avgG) + ", Green Standard Deviation: % s"  % (sd_G))
    print("Blue Mean: % s" % (avgB) + ", Blue Standard Deviation: % s"  % (sd_B))
    #Red Mean: 0.28799817, Red Standard Deviation: 0.37072635
    #Green Mean: 0.14884506, Green Standard Deviation: 0.19627
    #Blue Mean: 0.055384334, Blue Standard Deviation: 0.094425365

    R_AVG = 0.28799817
    R_SD = 0.37072635
    R_L = R_AVG - R_SD
    R_R = R_AVG + R_SD
    G_AVG = 0.14884506
    G_SD = 0.19627
    G_L = G_AVG - G_SD
    G_R = G_AVG + G_SD
    B_AVG = 0.055384334
    B_SD = 0.094425365
    B_L = B_AVG - B_SD
    B_R = B_AVG + B_SD

    goodimgsSC = []
    badimgsSC = []
    goodlabelsSC = []
    badlabelsSC = []
    for i in range(len(images)):
        imageR = images[i,:,:,0]
        imageG = images[i,:,:,1]
        imageB = images[i,:,:,2]
        
        imageR = imageR.flatten()
        imageG = imageG.flatten()
        imageB = imageB.flatten()
        
        avg_R = np.average(imageR)
        avg_G = np.average(imageG)
        avg_B = np.average(imageB)

        if (avg_R >= R_L and avg_R <= R_R) and (avg_G >= G_L and avg_G <= G_R) and (avg_B >= B_L and avg_B <= B_R):
            goodimgsSC.append(images[i])
            print("good image appended")
            goodlabelsSC.append(labels[i])
        else: 
            badimgsSC.append(images[i])
            print("bad image appended")
            badlabelsSC.append(labels[i])

    #len(goodimgsSC)
    #86420
    #len(badimgsSC)
    #830

    plt.figure()
    fig, axes = plt.subplots(2,10, dpi=300)
    axes = np.hstack(axes)

    for i in range(0,10):
        axes[i].imshow(goodimgsSC[i])

    for i in range(10,20):
        axes[i].imshow(badimgsSC[i])

    plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "test9.png")

#Now take mean and standard deviation of our new 'good'  (goodimgsSC[9]) image to filter all images off of
img = goodimgsSC[9] #from running on all images mean/standard deviation filter
imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]

imgR = imgR.flatten()
imgG = imgG.flatten()
imgB = imgB.flatten()

avgR = np.average(imgR)
avgG = np.average(imgG)
avgB = np.average(imgB)

#Check standard devaition from mean of each color value
sd_R = np.std(imgR)
sd_G = np.std(imgG)
sd_B = np.std(imgB)

print("Red Mean: % s" % (avgR) + ", Red Standard Deviation: % s"  % (sd_R))
print("Green Mean: % s" % (avgG) + ", Green Standard Deviation: % s"  % (sd_G))
print("Blue Mean: % s" % (avgB) + ", Blue Standard Deviation: % s"  % (sd_B))
#Red Mean: 0.4296188, Red Standard Deviation: 0.4651832
#Green Mean: 0.1810814, Green Standard Deviation: 0.20051083
#Blue Mean: 0.04572032, Blue Standard Deviation: 0.057372868

#Switch to half a standard deviation try and filter out >= 20,000 images
R_AVG = 0.42961887
#R_SD = 0.4651832
R_SD = 0.2325916
R_L = R_AVG - R_SD
R_R = R_AVG + R_SD
G_AVG = 0.1810814
#G_SD = 0.20051083
G_SD = 0.100255415
G_L = G_AVG - G_SD
G_R = G_AVG + G_SD
B_AVG = 0.04572032
#B_SD = 0.057372868
B_SD = 0.028686434
B_L = B_AVG - B_SD
B_R = B_AVG + B_SD

goodimgsSC = []
badimgsSC = []
goodlabelsSC = []
badlabelsSC = []
for i in range(len(images)):
    imageR = images[i,:,:,0]
    imageG = images[i,:,:,1]
    imageB = images[i,:,:,2]
    
    imageR = imageR.flatten()
    imageG = imageG.flatten()
    imageB = imageB.flatten()
    
    avg_R = np.average(imageR)
    avg_G = np.average(imageG)
    avg_B = np.average(imageB)

    if (avg_R >= R_L and avg_R <= R_R) and (avg_G >= G_L and avg_G <= G_R) and (avg_B >= B_L and avg_B <= B_R):
        goodimgsSC.append(images[i])
        print("good image appended")
        goodlabelsSC.append(labels[i])
    else: 
        badimgsSC.append(images[i])
        print("bad image appended")
        badlabelsSC.append(labels[i])

#len(goodimgsSC)
#44765
#len(badimgsSC)
#43485

plt.figure()
fig, axes = plt.subplots(2,10, dpi=300)
axes = np.hstack(axes)

for i in range(0,10):
    axes[i].imshow(goodimgsSC[i])

for i in range(10,20):
    axes[i].imshow(badimgsSC[i])

plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "test12.png")

#write goodimgsSC to file
newImg = np.array(goodimgsSC)
newLabel = np.array(goodlabelsSC)

np.save("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/retina_images_good"+ ".npy", newImg)
np.save("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/labels_good"+".npy", newLabel)

#subset diagnosis with new good labels 
#done in r