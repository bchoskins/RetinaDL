#Age Prediction
import pandas as pd 
import numpy as np
import keras
import random
import itertools
import statistics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from skimage.exposure import equalize_hist
from scipy.stats import pearsonr
import seaborn as sns; sns.set(color_codes=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

images = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/retina_images_good.npy")
xtrainLen = len(images)
images = np.divide(images, images.max(axis=(1,2,3), keepdims=True))

#shows dimensions of person 10, all pixels by all pixels and red RGB value (1=red,2=green,3=blue)
#images.shape
#images[10,:,:,1].shape

diagnosis = pd.read_csv('/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/DiagnosisWithAge.csv', index_col=0)
#diagnosis.head()
#len(diagnosis)

#order of IDs matches images already
#####take age instead of depression diagnosis
y = diagnosis.age

########
random.seed(4)
ind = np.arange(len(y))
np.random.shuffle(ind)
#train = 0, validate = 1, test = 2
ind = np.array((ind[0:27000], ind[27000:35000], ind[35000:]))
images[ind[0]].shape


#Model
keras.backend.clear_session()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=512,kernel_size=(2,2),activation='relu'))
model.add(Conv2D(filters=1024,kernel_size=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())

model.add(Dense(1,activation='linear'))

adam = keras.optimizers.Adam(lr=1e-2)

model.compile(loss='mean_squared_error', metrics=['accuracy'],optimizer='adam')

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=0, mode='auto')

cp = keras.callbacks.ModelCheckpoint(filepath="/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/ModelAge.h5",
        verbose=1, save_best_only=True)

model.fit(images[ind[0]], y.iloc[ind[0]], batch_size=16, callbacks=[es,cp],epochs=30, validation_data=[images[ind[1]], y.iloc[ind[1]]])

#model = keras.models.load_model("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model2.h5")

pred = model.predict(images[ind[2]])

true = y.iloc[ind[2]]

#Age stats
pearsonr(true, pred[:,0])
statistics.mean(abs(true-pred[:,0]))

#plot true vs predicted cbornreg plot
sns.regplot(x=true, y=pred[:,0])

plt.show()

plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/" + "regplot.png")

from scipy.stats import ttest_ind
#Depression stats
dep = diagnosis.Depression == 1
depDiagnosed = diagnosis[dep]
depAge = depDiagnosed["age"] #use as true
pred2 = model.predict(images[dep])
pearsonr(depAge, pred2[:,0])
#(0.7575491745369162, 0.0)
statistics.mean(abs(depAge-pred2[:,0]))
#3.9947814276857003
ttest_ind(abs(depAge-pred2[:,0]),abs(true-pred[:,0]))
#Ttest_indResult(statistic=-3.594838820421356, pvalue=0.00032571635628506827)
#Depression has a significance on predicting age

#Panic Attack stats
panic = diagnosis['Panic.Attacks'] == 1
panicDiagnosed = diagnosis[panic]
panicAge = panicDiagnosed["age"]
pred3 = model.predict(images[panic])
pearsonr(panicAge, pred3[:,0])
#(0.7519279863057625, 6.586869712776213e-201)
statistics.mean(abs(panicAge-pred3[:,0]))
#4.038258927085183
ttest_ind(abs(panicAge-pred3[:,0]),abs(true-pred[:,0]))
#Ttest_indResult(statistic=-1.639602116755517, pvalue=0.1011168784350734)

controls = diagnosis.Depression == 0
ctrlDiagnosed = diagnosis[controls]
ctrlAge = ctrlDiagnosed["age"]
predCtrl = model.predict(images[controls])
np.mean(abs(ctrlAge-predCtrl[:,0]))
#4.136008420828924