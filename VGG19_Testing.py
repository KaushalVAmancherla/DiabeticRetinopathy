import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)

img_width=224
img_height=224

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

from keras.models import load_model
vgg_binary = load_model('vgg19_binary.h5')

from PIL import Image

test_img = image.load_img('fullKaggle/test/2a08ed6bbcbc.png')
test_img = test_img.resize((224,224), resample=Image.BICUBIC)
test_arr = image.img_to_array(test_img)
test_arr = np.expand_dims(test_arr, axis = 0)
test_arr /= 255.0

plt.imshow(test_img)
score = float(vgg_binary.predict(test_arr))

if score < 0.5:
    print("Healthy")
else:
    print("Diabetic")

import anvil.server

anvil.server.connect("TQ7Y7WERTJCHJJAJBWLYD4Z2-GHU2NNXFID5HNH7D")

import anvil.media
from PIL import Image

@anvil.server.callable
def classify_image(file):
    with anvil.media.TempFile(file) as filename:
        img = image.load_img(filename)
        
    img = img.resize((224,224), resample=Image.BICUBIC)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    arr /= 255.0
    
    score = vgg_binary.predict(arr)
    
    return ('Healthy' if score < 0.5 else 'Diabetic', float(score))