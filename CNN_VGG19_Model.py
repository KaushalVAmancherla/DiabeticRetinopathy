# Import the necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)

# Setting the dimensions of our images.

img_width, img_height = 224, 224

train_data_dir = 'fullKaggle/train'
validation_data_dir = 'fullKaggle/validation'

nb_train_samples = 2662
nb_validation_samples = 500

epochs = 35   
batch_size = 20

# Setting the input shape format: 3 is the color channels (RGB)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

vgg = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)        # print out the model summary

# Freeze the layers so that they are not trained during model fitting. We want to keep the imagenet weights
for layer in vgg.layers: 
    layer.trainable=False

# Change the final dense layer to 1 node (sigmoid activation) for binary classification
x = vgg.layers[-2].output
output_layer = Dense(1, activation='sigmoid', name='predictions')(x)

# Combine the output layer to the original model
vgg_binary = Model(inputs=vgg.input, outputs=output_layer)

# Sanity check: Print out the model summary. The final layer should have 1 neuron only (again, using sigmoid activation)
vgg_binary.summary()

from keras import optimizers
sgd = optimizers.SGD(lr=0.005, decay=1e-6,momentum=0.9,nesterov=True)

vgg_binary.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Defining Image transformations: normalization (rescaling) for both training and testing images
# Defining Image transformations: Augmenting the training data with the following transformations 
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Setting up the flow of images in batches for training and validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Printing out the class labels for both training and validation sets
print(train_generator.class_indices)
print(validation_generator.class_indices)

# Fitting the modified vgg19 model on the image batches set up in the previous step
# Save the model (full model). Save the training history
history = vgg_binary.fit_generator(
        train_generator,
        steps_per_epoch=2662 // batch_size,
        epochs=epochs,                           
        validation_data=validation_generator,
        validation_steps=500 // batch_size)

vgg_binary.save('vgg19_binary.h5')
print("Saved vgg19 model to disk") # the modlsize is over 500MB