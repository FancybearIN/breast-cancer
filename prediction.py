from tkinter import *
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set image data format
K.set_image_data_format('channels_last')

model = load_model('14090finalmodel.h5')

while True:
    path = filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
    img = cv2.imread(path)
    plt.imshow(img)
    plt.show()
    test_image = cv2.imread(path)
    test_image = cv2.resize(test_image, (140, 92))
    test_image = test_image.reshape(92, 140, 3)
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    test_image = np.expand_dims(test_image, axis=0)
    pa = model.predict(test_image)
    if np.argmax(model.predict(test_image), axis=-1) == [0]:
        s = "BENIGN with Accuracy: " + str(pa[0][0] * 100) + "%\n"
    else:
        s = "MALIGNANT with Accuracy: " + str(pa[0][1] * 100) + "%\n"

    print(s)