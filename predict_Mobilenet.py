import keras
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
#import sys
#from PIL import Image

from keras.preprocessing import image

#import matplotlib.pyplot as plt

import numpy as np
import os

path = 'test\\'
folders = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for folder in d:
        folders.append(os.path.join(r, folder))

files = []
			
for path_d in folders:			
    for r, d, f in os.walk(path_d):
        for file in f:
            if '.jpg' in file:#file.endswith(".jpg")
                files.append(os.path.join(r, file))


	
#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())train_generator.class_indices return a dicionary name and an index,
# with name is dogs, cats, ...; index is index of dogs, cats, ... example: dogs have index is 0, cats have index is 5, ...

labels = ['cats','daisy','dandelion','dogs','horses','rose','sunflower','tulip']
#model = load_model("model\\alpha_InceptionResNetV2_")#wrong on windows, right on linux
#model = load_model("model\\alpha_InceptionV3_")#ok on windows
model = load_model("model\\alpha_MobileNet_")
#model = load_model("model\\alpha_MobileNet_None_")

#'alpha_MobileNet_'
# print(model.summary())

total = len(files)
numTrue = 0

fw = open("result\\weight_alpha_MobileNet_result.txt", "a")

print("\n")
for f in files:
    img = image.load_img(f, target_size = (224, 224))#Image.open(f)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
	
    ypred = model.predict(x)[0]#model.predict(np.asarray([x]))
    y_prex_index = np.argmax(ypred)
    predict_name = labels[y_prex_index]
	
    tmp_arr = f.split('\\')
    name = tmp_arr[1]
    if(name == predict_name):
        numTrue += 1
        content = "True: " + f + " -- " + labels[y_prex_index] + "\n"
        print(content)
        fw.write(content)
    else:
        content = "False: " + f + " -- " + labels[y_prex_index] + "\n"
        print(content)
        fw.write(content)
		
#result = "total = " + total + " numTrue = " + numTrue + " alpha_InceptionResNetV2_ have acc = total/numTrue = " + numTrue/total
#result = "total = " + total + " numTrue = " + numTrue + " alpha_InceptionV3_ have acc = total/numTrue = " + numTrue/total
result = "total = " + str(total) + " numTrue = " + str(numTrue) + " alpha_MobileNet_ have acc = numTrue/total = " + str(numTrue/total)
print(result)
fw.write(result)
fw.close()
	


