import pandas as pd 
import numpy as np
import keras
import random
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.exposure import equalize_hist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

images = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/retina_images_good.npy")
xtrainLen = len(images)
images = np.divide(images, images.max(axis=(1,2,3), keepdims=True))
#images2 = images.reshape(xtrainLen, 128, 128,3)

#shows dimensions of person 10, all pixels by all pixels and red RGB value (1=red,2=green,3=blue)
images.shape
images[10,:,:,1].shape

diagnosis = pd.read_csv('/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/newDiagnosis.csv', index_col=0)
diagnosis.head()
len(diagnosis)

#order of IDs matches images already
y = diagnosis.Depression

#Create a test set and validation set
        #random.seed(4)
        #ind = np.arange(len(y))
        #np.random.shuffle(ind)
        #train = 0, validate = 1, test = 2
        #ind = np.array((ind[0:27000], ind[27000:35000], ind[35000:]))
        #images[ind[0]].shape
        #(27000, 128, 128, 3)

        #np.save("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/train_test_index.npy", ind)
        ind = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/train_test_index.npy", allow_pickle=True)

#under sample to deal with class imbalance (50% of majority/minority class)
        # Class count
        count_class_0, count_class_1 = y.value_counts()
        #count_class_0 = 40632
        #count_class_1 = 4133
        y_class_0 = y[y == 0]
        y_class_1 = y[y == 1]

        y_class_0_under = y_class_0.sample(count_class_1)
        y_under = pd.concat([y_class_0_under, y_class_1], axis=0)

        random.seed(4)
        index = np.arange(len(y_under))
        np.random.shuffle(index)
        #train = 0, validate = 1, test = 2
        index = np.array((index[0:5000], index[5000:6500], index[6500:]))
        images[index[0]].shape

        #up sample to deal with class imbalance (50% of majority/minority class)
        y_class_1_over = y_class_1.sample(count_class_0, replace=True)
        y_over = pd.concat([y_class_0, y_class_1_over], axis=0)

        random.seed(4)
        idx = np.arange(len(y_over))
        np.random.shuffle(idx)
        #train = 0, validate = 1, test = 2
        idx = np.array((idx[0:50000], idx[50000:65000], idx[65000:]))
        images[idx[0]].shape

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

#now have up sampled pos class indeces in random order, still only want 25,000 train, 8,000 validate, 7,632 test 
#in total with controls though

        y_trainD = random.sample(y_depressed_train_up, 12500) 
        y_trainC = random.sample(list(y_control[0]), 12500)
        y_train = y_trainD + y_trainC
        np.random.shuffle(y_train)

        y_validateD = random.sample(y_depressed_validate_up, 4000) 
        y_validateC = random.sample(list(y_control[1]), 4000)
        y_validate = y_validateD + y_validateC
        np.random.shuffle(y_validate)

        y_testD = random.sample(y_depressed_test_up, 3816) 
        y_testC = random.sample(list(y_control[2]), 3816)
        y_test = y_testD + y_testC
        np.random.shuffle(y_test)

#Model with bad images not included 
        #model = Sequential()
        #odel.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        #model.add(Flatten())
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(1, activation='sigmoid',))

        #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        #es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=0, mode='auto')

        #cp = keras.callbacks.ModelCheckpoint(filepath="/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model2.h5",
                #verbose=1, save_best_only=True)

        #class_weights = {0:1.1, 1:10.95}
        #model.fit(images[ind[0]], y[ind[0]], batch_size=128, callbacks=[es,cp],epochs=30, class_weight = class_weights, validation_data=[images[ind[1]], y[ind[1]]])

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

model.add(Dense(1,activation='sigmoid',bias_initializer=keras.initializers.Constant(value=0)))

#adam = keras.optimizers.Adam(lr=1e-6)
adam = keras.optimizers.Adam(lr=1e-9)

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=0, mode='auto')

cp = keras.callbacks.ModelCheckpoint(filepath="/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model2.h5",
        verbose=1, save_best_only=True)

#class_weights = {0:1.986, 1:2.015}
#1.1034820990681706
#10.663507109004739
#class_weight = class_weights
model.fit(images[y_train], y.iloc[y_train], batch_size=64, callbacks=[es,cp],epochs=3000, validation_data=[images[y_validate], y.iloc[y_validate]])

#image check
        plt.figure()
        fig, axes = plt.subplots(1,10, dpi=300)
        axes = np.hstack(axes)

        for i in range(0,10):
        axes[i].imshow(images[i])

        plt.savefig("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/imageCheck.png")

#checks
        sum(np.isnan(y.iloc[y_train]))
        sum(np.isnan(y.iloc[y_validate]))

        sum(y.iloc[y_train]) = 12500
        sum(y.iloc[y_validate]) = 4000
        sum(y.iloc[y_test]) = 3816

#model = keras.models.load_model("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model2.h5")

pred = model.predict(images[y_test])

#evl = model.evaluate(images[ind[2]], y[ind[2]])

true = y.iloc[y_test]

#AROC
roc = roc_auc_score(true, pred)