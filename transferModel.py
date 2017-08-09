# This file import createTransferCodes file
# createTransferCodes does following things:
# 1. download the VGG16 model data (.npy file)
# 2. apply transformations if required
# 3. create transfer codes and store them in the project directory

import numpy as np
import csv
import dataHandler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as K

EPOCHS = 20
optimizer = Adam(lr=0.01)
create_new_transfer_codes = True


# ---------------------------------------------------
if create_new_transfer_codes:
    print('Creating new transfer codes for data...')
    import createTransferCodes

# ---------------------------------------------------

# load files
with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader]).squeeze()
    labels = labels[:-1]
    print('loaded labels', labels.shape)

with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))
    print('loaded codes', codes.shape)


# -------------------------------------------------------------
# split data
from sklearn.model_selection import train_test_split
labels, classes = dataHandler.one_hot_encode(labels)
X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.20, random_state=42)
X_train = X_train.astype('float32')
print('X shape', X_train.shape)
print('y shape', y_train.shape)

# --------------------------------------------------------------
lr_scheduler = LearningRateScheduler(dataHandler.scheduler)
K.set_image_dim_ordering('th')

# create model
model = Sequential()
model.add(Dense(512, input_shape=(4096,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))


# Compile and fit model
# LR scheduler added which updates LR every 10 epochs
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, nb_epoch=EPOCHS, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# predict
result = model.predict(X_test)
print('predicted', np.argmax(result, axis=1))
print('actual', np.argmax(y_test, axis=1))

