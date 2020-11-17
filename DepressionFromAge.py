#Keeping Age as a feature for predicting Depression
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
y = diagnosis.Depression

########
y_depressed = np.where(y==1)
y_control = np.where(y==0)

random.seed(4)
np.random.shuffle(y_depressed[0])
#train = 0, validate = 1, test = 2
y_depressed = np.array((y_depressed[0][0:2500], y_depressed[0][2500:3500], y_depressed[0][3500:]))

random.seed(4)
np.random.shuffle(y_control[0])
#train = 0, validate = 1, test = 2
y_control = np.array((y_control[0][0:25000], y_control[0][25000:33000], y_control[0][33000:]))

y_depressed_train_up = list(itertools.chain.from_iterable(itertools.repeat(i, 10) for i in y_depressed[0])) 
y_depressed_validate_up = list(itertools.chain.from_iterable(itertools.repeat(i, 8) for i in y_depressed[1])) 
y_depressed_test_up = list(itertools.chain.from_iterable(itertools.repeat(i, 13) for i in y_depressed[2])) 
y_depressed_test_up = y_depressed_test_up[0:7632]

#shuffle
np.random.shuffle(y_depressed_train_up)
np.random.shuffle(y_depressed_validate_up)
np.random.shuffle(y_depressed_test_up)

y_train = list(y_control[0]) + y_depressed_train_up
y_validate = list(y_control[1]) + y_depressed_validate_up
y_test = list(y_control[2]) +  y_depressed_test_up

np.random.shuffle(y_train)
np.random.shuffle(y_validate)
np.random.shuffle(y_test)

#Model
keras.backend.clear_session()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())

#model.add(Conv2D(filters=512,kernel_size=(2,2),activation='relu'))
#model.add(Conv2D(filters=1024,kernel_size=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.))
model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))

adam = keras.optimizers.Adam(lr=1e-6)

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=0, mode='auto')

cp = keras.callbacks.ModelCheckpoint(filepath="/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/ModelDepression_Age.h5",
        verbose=1, save_best_only=True)

model.fit(images[y_train], y.iloc[y_train], batch_size=256, callbacks=[es,cp],epochs=30, validation_data=[images[y_validate], y.iloc[y_validate]])

#model = keras.models.load_model("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model2.h5")

pred = model.predict(images[y_test])

true = y.iloc[y_test]

#Depression stats
#AROC
roc = roc_auc_score(true, pred)
#0.564905960476466