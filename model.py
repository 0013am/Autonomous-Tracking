import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from keras.layers import MaxPooling2D, Dropout
from keras.utils import print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def keras_model(image_x,image_y):
    model=Sequential()
    model.add(Lambda(lambda x:x/127.5 -1.,input_shape=(image_x,image_y,1)))
    model.add(Conv2D(24,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),padding='same'))

    model.add(Conv2D(36,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),padding='same'))

    model.add(Conv2D(48,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),padding='same'))

    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),padding='same'))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

   model.compile(optimizer='adam',loss='mse')
   filepath = "Autopilot_10.h5"
   checkpoint = ModelCheckpoint(filepath,verbose=1,save_best_only=True)
   callbacks_list=[checkpoint]

   def loadFromPickle():
    with open("features", "rb") as f:
        features=np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels=np.array(pickle.load(f))
    return features, labels

    def main():
        features,labels=loadFromPickle()
        features,labels=shuffle(features,labels)
        train_x,test_x,train_y,test_y=train_test_split(features,labels,random_state=0,test_size=0.3)
        train_x = train_x.reshape(train_x.shape[0],66,200,1)
        test_x = test_x.reshape(test_x.shape[0],66,200,1)
        model,callbacks_list=keras_model(66,200)
        model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=3,batch_size=32,callbacks=callbacks_list)
        print_summary(model)
        model.save(filepath)
    main()
