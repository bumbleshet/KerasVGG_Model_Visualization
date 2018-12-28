# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:57:16 2018

@author: ldhandu
"""

from keras.applications import vgg16
import keras.backend as K
from keras.preprocessing import image

from keras.activations import linear
import matplotlib.pyplot as plt
from keras import models
import numpy as np
import cv2
import PIL as p

# MODIFIED THE MODEL AND SAVED TO THE DISK FRO TRANSFER LEARNING
#vgg_model = vgg16.VGG16()
#
#
##for layer in vgg_model.layers:
##    if hasattr(layer,'activation'):
##        layer.activation = linear
#        
#vgg_model.save('D:/learning/My_custom_ML_models/vgg_tl_model_rgb'+'.h5')

#image for visualizing and preprocessing
test_img = image.load_img('C:/Users/ldhandu/Desktop/car.jpg',grayscale=False,target_size=(224,224,3))
#plt.imshow(test_img)

img_arr = image.img_to_array(test_img)
print('original image ',img_arr.shape)

#exp_img = np.expand_dims(img_arr,axis=0)
#print('batch image',exp_img.shape)

img_batch = np.array([img_arr], dtype=np.float32)
print('batch image',img_batch.shape)

load_vgg_model = models.load_model('D:/learning/My_custom_ML_models/vgg_tl_model_rgb.h5')
#load_vgg_model.summary()

conv_layer = load_vgg_model.layers[5]
print(conv_layer)

#function that returns the output after each conv operation
output_fn = K.function([load_vgg_model.input],[conv_layer.output])
output = output_fn([img_batch])[0] # yet to get this step

output_arr = np.asarray(output)
#otput is an array
print('output',output_arr.shape)

out_squeeze = np.squeeze(output_arr,axis=0)
print('output sqz ', out_squeeze.shape)

img_moved_axis = np.moveaxis(out_squeeze,2,0)
print('moved axis',img_moved_axis.shape)

plt.axis("off")
#when cmap is set to gray plt.imshow accepts images  w*h else we have to input w*h*d
plt.imshow(img_moved_axis[12])




