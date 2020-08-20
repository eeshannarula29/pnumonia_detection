import numpy as np
from keras.models import load_model

from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk
import cv2
import os

from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def edit_photo(path):
    img = cv2.resize(cv2.imread(path),(224,224))
    return img.reshape(1,224,224,3)

def getprediction(a):
    i = list(a[0]).index(max(list(a[0])))
    if i == 0:
        return 'NORMAL'
    elif i == 1:
        return 'PNEUMONIA'

def build_model():
    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, outputs=x)
    return model



model = build_model()
model.load_weights('output.h5')


root = Tk()

PATH = StringVar()
CATIGORY = StringVar()

def browsefunc():
    PATH.set(askopenfilename())
    img = Image.open(PATH.get())
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

def predict():
    if PATH.get() is not None:
       prediction = getprediction(model.predict(edit_photo(PATH.get())))
       CATIGORY.set('prediction is : ' + prediction)

text1 = Text(root)

text1.insert(INSERT, "Hello.....")
text1.insert(INSERT, "                                                                      ")
text1.insert(INSERT, "this software is made to detect Pneumomia")
text1.insert(INSERT, "                                       ")
text1.insert(INSERT, "All you have to do is ckick on 'choose a image' and select your xray")
text1.insert(INSERT, "            ")
text1.insert(INSERT, "then click on 'predict the image' and you will get the results")

text1.grid(row = 0,column = 0)

panel = Label(root)
panel.grid(row = 1,column = 0)

browsebutton = Button(root, text="choose a image", command=browsefunc)
browsebutton.grid(row = 2,column=0)

predictbutton = Button(root, text="predict the image", command=predict)
predictbutton.grid(row = 3,column=0)

prediction = Label(root, textvariable=CATIGORY)
prediction.grid(row = 4,column=0)

mainloop()
