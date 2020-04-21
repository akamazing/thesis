
# coding: utf-8

# In[ ]:


from __future__ import division, print_function, absolute_import

# Import MNIST data
import os
import enum
import pickle
import logging
import datetime
import shutil, sys                                                                                                                                                    
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from keras import backend as K
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#from keras.datasets import mnist
#from keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.metrics import multilabel_confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data

K.set_image_dim_ordering('th')
logging.getLogger("tensorflow").setLevel(logging.INFO)

# In[ ]:
Shade = "Shade"
Rotate = "Rotate"
Shear = "Shear"
ShiftX = "ShiftX"
ShiftY = "ShiftY"
ZoomX = "ZoomX"
ZoomY = "ZoomY"

Test = "Test"
Train = "Train"

# Training Parameters
learning_rate = 0.01
num_steps = 500
batch_size = 128

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
dropout = 0.25 # Dropout, probability to drop a unit

num_classes = 26
num_images = 1000

NN = "NN"
CNN = "CNN"
KNN = "KNN"
NB = "NB"
SVM = "SVM"

Algo = CNN
