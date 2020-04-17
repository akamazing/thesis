# Import common modules

from __future__ import division, print_function, absolute_import

import os
import enum
import pickle
import logging
import datetime
import shutil, sys                                                                                                     
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import tensorflow as tf
from random import randrange                                  
from matplotlib import pyplot
from itertools import groupby
from operator import itemgetter
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data

K.set_image_dim_ordering('th')

# Set logging info
logging.getLogger("tensorflow").setLevel(logging.INFO)

# Metamorphic transformations name
Shade = "Shade"
Rotate = "Rotate"
Shear = "Shear"
ShiftX = "ShiftX"
ShiftY = "ShiftY"
ZoomX = "ZoomX"
ZoomY = "ZoomY"

# Model mode
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

# Number of and classes
num_classes = 10
# Number of images to be sampled
num_images = 1000

# Machine learning algorithms
NN = "NN"
CNN = "CNN"
KNN = "KNN"
NB = "NB"
SVM = "SVM"

# Datasets
Digit = "digit"
Letter = "letter"
Fashion = "fashion"

# Set default algo
Algo = CNN

