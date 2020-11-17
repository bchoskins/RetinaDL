# Steps would look like this:
# 1) process phenotypes (DONE)
# 2) fit models to depression, anxiety in CV
# 3) create saliency maps (https://github.com/raghakot/keras-vis/blob/master/vis/visualization/saliency.py)
# 4) create table describing cohort
# 5) make figures from above & write paper

#Read in Mental_Health_Results.R from Mental_Health_Diagnosis.R
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

#data = pd.read_csv("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Mental_Health_Results.csv") 

#print(data.head())

#labels = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/labels_full.npy")
#labels.shape

#Load in retina image data
images = np.load("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/data_prep/npy_files/retina_images_full.npy")
xtrainLen = len(images)
images = np.divide(images, images.max(axis=(1,2,3), keepdims=True))
images = images.reshape(xtrainLen, 128, 128,3)

#shows dimensions of person 10, all pixels by all pixels and red RGB value (1=red,2=green,3=blue)
images.shape
images[10,:,:,1].shape

#Read in merged diagnosis data
diagnosis = pd.read_csv('/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Diagnosis.csv', index_col=0)
diagnosis.head()

#
y = diagnosis.Depression

#Create validation set
X_train, X_test, y_train, y_test = train_test_split(images, y, random_state=42, test_size=0.2)

#Initial model with bad images included 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=0, mode='auto')

cp = keras.callbacks.ModelCheckpoint(filepath="/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model1.h5",
        verbose=1, save_best_only=True)


model.fit(X_train, y_train, batch_size=128, callbacks=[es,cp],epochs=30, validation_data=(X_test, y_test))


model = keras.models.load_model("/Dedicated/jmichaelson-wdata/rotating_students/bhoskins/RetinaDL/Model1.h5")

pred = model.predict(images[0:999])

true = y[0:999]

#AROC
roc = roc_auc_score(true, pred)
#0.620342...

