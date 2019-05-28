
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint#

# In[2]:

model_file_prefix = "MobileNet_"
base_model = MobileNet(weights = 'imagenet', include_top = False) #imports the mobilenet model and discards the last 1000 neuron layer.
#base_model = MobileNet(weights = 'None', include_top = False)#just use architecture, don't use weights

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation = 'relu')(x) #dense layer 2
x = Dense(512, activation = 'relu')(x) #dense layer 3
preds = Dense(8, activation = 'softmax')(x) #final layer with softmax activation


# In[3]:


model = Model(inputs = base_model.input, outputs = preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:


#for layer in model.layers[:20]:
#    layer.trainable = False
	
#for layer in model.layers[20:]:
#    layer.trainable = True
	
#don't re-train base_model
for layer in base_model.layers:
	layer.trainable = False

print(model.summary())#list layers which were trained or not trained.

# In[5]:



#train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input) #included in our dependencies												 
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                    featurewise_center = False,
                                    samplewise_center = False,
                                    featurewise_std_normalization = False,
                                    samplewise_std_normalization = False,
                                    zca_whitening = False,
                                    rotation_range = 45,
                                    width_shift_range = 0.25,
                                    height_shift_range = 0.25,
                                    horizontal_flip = True,
                                    vertical_flip = False,
                                    channel_shift_range = 0.5,
                                    zoom_range = [0.5, 1.5],
                                    brightness_range = [0.5, 1.25],
                                    fill_mode = 'nearest') #included in our dependencies
									
train_generator = train_datagen.flow_from_directory('train/', # this is where you specify the path to the main data folder
                                                 target_size = (224, 224),
                                                 color_mode = 'rgb',
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle = True)
												 
 
# In[33]:


model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
#model.fit_generator(generator = train_generator,
#                   steps_per_epoch = step_size_train,
#                   epochs = 50)
		   
checkpoint = ModelCheckpoint(filepath = "model/alpha_" + model_file_prefix, verbose = 1, save_best_only = True)#save model	
				   
model.fit_generator(generator = train_generator,
                    steps_per_epoch = step_size_train,
                    epochs = 50,
                    validation_data = train_generator,
                    validation_steps = step_size_train,
                    callbacks = [checkpoint])


# Save model after training finished
#model.save('./model/model_MobileNet.h5')
