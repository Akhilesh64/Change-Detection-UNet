import numpy as np 
import os
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from getData import getData 

def unet(x):
    
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'block1_conv1')(x)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name = 'block1_pool')(conv1)
    
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'block2_conv1')(x)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name = 'block2_pool')(conv2)
    
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block3_conv1')(x)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block3_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name = 'block3_pool')(conv3)
    
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block4_conv1')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', , name = 'block4_conv2')(x)
    drop4 = Dropout(0.5, name = 'dr4')(x)
    x = MaxPooling2D(pool_size=(2, 2), name = 'block4_pool')(drop4)

    x = Conv2D(1024, 3, activation = 'relu', padding = 'same', name = 'block5_conv1')(x)
    x = Conv2D(1024, 3, activation = 'relu', padding = 'same', name = 'block5_conv2')(x)
    x = Dropout(0.5, name = 'dr5')(x)

    x = UpSampling2D(size = (2,2), name = 'up6')(x)
    x = Conv2D(512, 2, activation = 'relu', padding = 'same', name = 'block6_conv1')(x)
    x = concatenate([drop4, x], axis = 3)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', , name = 'block6_conv2')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block6_conv3')(x)

    x = UpSampling2D(size = (2,2), name = 'up7')(x)
    x = Conv2D(256, 2, activation = 'relu', padding = 'same', name = 'block7_conv1')(x)
    x = concatenate([conv3, x], axis = 3)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block7_conv2')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block7_conv3')(x)

    x = UpSampling2D(size = (2,2), name = 'up8')(x)
    x = Conv2D(128, 2, activation = 'relu', padding = 'same', name = 'block8_conv1')(x)
    x = concatenate([conv2,x], axis = 3)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'block8_conv2')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'block8_conv3')(x)

    x = UpSampling2D(size = (2,2), name = 'up9')(x)
    x = Conv2D(64, 2, activation = 'relu', padding = 'same', name = 'block9_conv1')(x)
    x = concatenate([conv1,x], axis = 3)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'block9_conv2')(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'block9_conv3')(x)
    x = Conv2D(2, 3, activation = 'relu', padding = 'same', name = 'block9_conv4')(x)
    x = Conv2D(1, 1, activation = 'sigmoid', name = 'block9_conv5')(x)

    return x


def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

net_input = Input(shape = (240,320,3)) 

model_output = unet(net_input)

model = Model(inputs = net_input, outputs = model_output)

opt = RMSprop(lr = 1e-4, rho=0.9, epsilon=1e-08, decay=0.)

model.compile(optimizer = opt, loss = loss, metrics = [acc])

early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=15, verbose=0, mode='auto')

redu = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, mode='auto')

# save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

data = getData()

model.fit(data[0], data[1], validation_split = 0.1, epochs = 500, batch_size = 1, callbacks = [early, redu], verbose=1, shuffle = True)
model.save('model.h5')
