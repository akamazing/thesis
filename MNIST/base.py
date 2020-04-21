from include import *

xTestBackup = []
yTestBackup = []
xTrainBackup = []
yTrainBackup = []

def conv_net(x_dict, n_classes, dropout, reuse, is_training):

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)

        # Convolution Layer with 64 filters
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):

    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    layer_3 = tf.layers.dense(layer_2, n_hidden_2)
    layer_4 = tf.layers.dense(layer_3, n_hidden_2)
    layer_5 = tf.layers.dense(layer_4, n_hidden_2)
    layer_6 = tf.layers.dense(layer_5, n_hidden_2)
    layer_7 = tf.layers.dense(layer_6, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_7, num_classes)
    return out_layer

# Define the model function (following TF Estimator Template)
def nn_model_fn(features, labels, mode):

    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


def reloadData(dataType):
    global xTestBackup
    global yTestBackup
    global xTrainBackup
    global yTrainBackup
    if dataType == "Test": # Reset test data
        global xTest
        global yTest
        xTest[:] = xTestBackup
        yTest[:] = yTestBackup
    elif dataType == "Train": # Reset training data
        global xTrain
        global yTrain
        xTrain[:] = xTrainBackup
        yTrain[:] = yTrainBackup

'''
Parameters:
    model:  The model to be trained
    xData: (List) List of test images
    yData: (List) List of test corresponding labels
    shuffle:(bool) Shuffle the data before testing
Returns a list of predictions for each data and the accuracy
'''
def TestModel(model, xData, yData, shuffle):
    # Prepare the input data
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': xData}, shuffle=shuffle)
    # Use the model to predict the images class
    preds = list(model.predict(input_fn))
    # calculate accuracy
    acc = 0.0
    for x in range(len(yData)):
        if (yData[x]==preds[x]):
            acc = acc+1
    acc = acc/len(yData)
    #return preds, accuracy['accuracy']
    return preds, acc

'''
Parameters:
    model:  The model to be trained
    xTrain: (List) List of images
    yTrain: (List) List of corresponding labels
    shuffle:(bool) Shuffle the data before training
'''
def TrainModel(model, xTrain, yTrain, shuffle):
    input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': xTrain}, y= np.array(yTrain),
            batch_size=batch_size, num_epochs=None, shuffle=shuffle)
    # Train the Model
    model.train(input_fn, steps=num_steps)

'''
Parameters:
    xData:  (List) List of images
    yData:  (List) List of corresponding labels
    tType:  (String) Metamorphic relation to use
    value:  (float) Amount of transformation to apply
    tranpose:   (bool) Apply a transpostion to final image
Returns a transformed list of images
'''

def Transform( xData, yData, tType, value, transpose ):

    n_images = len(xData)
    xTemp = []
    yTemp = []

    datagen = ImageDataGenerator()# fit parameters from data
    #  tType = Shade
    if tType == "Shade":
        Xnew = [[[[v-value if v-value>0.0 else 0.0 for v in n] for n in x[0]]] for x in xData]
        # Xnew = [[[[v-value for v in n] for n in x[0]]] for x in xData]
        Xnew = np.array(Xnew)
        xTemp[:] = Xnew.astype('float32')

    # tType = Rotate
    if tType == "Rotate":
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'theta':value})
            xTemp.append(x)

    # tType = Shear
    if tType == "Shear":
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'shear':value})
            xTemp.append(x)

    # tType = ShiftX
    if tType == "ShiftX":
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'tx':value})
            xTemp.append(x)

    # tType = ShiftY
    if tType == "ShiftY":
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'ty':value})
            xTemp.append(x)

    # tType = ZoomX
    if tType == "ZoomX":
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'zx':value})
            xTemp.append(x)

     # tType = ZoomY
    if tType == "ZoomY":
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'zy':value})
            xTemp.append(x)

    # Generate final list
    xTmp = []
    for x in xTemp:
        if transpose:
            xTmp.append(np.transpose(x))
        else:
            xTmp.append(x)
    xTmp = np.array(xTmp)
    return xTmp

'''
Parameters:
    xTest:  (List) List of images
    yTest:  (List) List of corresponding labels

Displays data in a nice image
'''
def DisplayData( xTest, yTest ):
    for i in range(len(xTest)):
        plt.imshow(np.reshape(xTest[i], [28, 28]), cmap='gray')
        plt.show()
        print ("Image Label: ", yTest[i])

'''
File Handler: opens a file
Location: location of file
name: name of file
Returns file pointer
'''
def OpenFile(Location, name):
    filename = str(Location)+"/"+str(name)
    fp = open(filename, 'w')
    return fp

'''
File Handler: closes a file
fp: file pointer
'''
def CloseFile(fp):
    fp.close()


'''
Parameters:
    predMatrix:  A list of predictions [num of instances][num_class][num_class]
    yMatrix:     A list of corresponding correct labels [num of instances][num_class][num_class]

Returns accuracy of prediction
'''
def getPredictionAccuracy (predMatrix, yMatrix):
    totalData = [0]*num_classes
    # Find total number of data in a class
    for y in yMatrix[0]:
        totalData[y] = totalData[y]+1

    aMatrix = []

    for i in range(len(predMatrix)):
        # Temp variable of size [num_class][num_class] to store data
        accMatrix = [[0]*num_classes for x in range(num_classes)]
        # increases the value in matrix by 1 for every correct prediction
        for j in range(len(predMatrix[i])):
            accMatrix[yMatrix[i][j]][predMatrix[i][j]] += 1
        # Accuracy = total correct predictions/total number of predictions
        for x in range(len(accMatrix)):
            for y in range(len(accMatrix[x])):
                accMatrix[x][y] = round(accMatrix[x][y]/totalData[x], 3)

        aMatrix.append(accMatrix)
    # Returns the prediction matrix
    return aMatrix

'''
Parameters:
    MT:  Metamorphic property
    accMatrix:  list of accuracy for each dataset

Displays accuracy in a nice plot
'''
def getDatasetAccuracyPlot(MT, accMatrix):
    figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    m = len(accMatrix)
    plt.axis([-m/2, m/2, 0, 1])
    plt.plot(np.arange(-m/2,m/2,1),accMatrix)
    plt.xlabel("Value")
    if (MT==Rotate or MT==Shear):
        plt.xlabel("Angle")
    if (MT==ShiftX or MT==ShiftY):
        plt.xlabel("Pixel")
    plt.ylabel("Accuracy")
    #plt.suptitle("MR: "+str(MT))
    return plt

'''
Parameters:
    MT:  Metamorphic property
    accMatrix:  list of accuracy for each dataset

Displays accuracy of each class in a nice plot
'''
def getClassByAccuracyPlot(MT, accMatrix):
    figure(num=None, figsize=(26, 12), dpi=180, facecolor='w', edgecolor='k')
    m = len(accMatrix)
    for j in range(num_classes):
        zero = []
        for i in range(len(accMatrix)):
            zero.append(accMatrix[i][j][j])
        plt.axis([-m/2, m/2, 0, 1])
        plt.subplot(2,5,j+1)
        plt.plot(np.arange(-m/2,m/2,1),zero)
        plt.xlabel("Value")
        if (MT==Rotate or MT==Shear):
            plt.xlabel("Angle")
        if (MT==ShiftX or MT==ShiftY):
            plt.xlabel("Pixel")
        plt.ylabel("Accuracy")
        plt.title('Digit: '+str(j))
    #plt.suptitle("MR: "+str(MT))
    plt.show()
    return plt

'''
Parameters:
    xTest:  (List) List of images
    yTest:  (List) List of corresponding labels

Displays data in a nice image
'''
def getMisclassificationPlot(MT, accMatrix):
    m = len(accMatrix)
    for j in range(num_classes):
        figure(num=None, figsize=(28, 8), dpi=180, facecolor='w', edgecolor='k')
        for k in range(0,num_classes):
            zero = []
            if j != k:
                for i in range(len(accMatrix)):
                    zero.append(accMatrix[i][j][k])
            else:
                continue
            plt.subplot(2,5,k+1)
            plt.axis([-m/2, m/2, 0, 1])
            plt.plot(np.arange(-m/2,m/2,1),zero)
            plt.xlabel("Value")
            if (MT==Rotate or MT==Shear):
                plt.xlabel("Angle")
            plt.ylabel("Misclssified to %s" %str(k))
        #plt.suptitle("Misclassification graph for %d" %(j))
        plt.show()
        print("---------------------------------------------------------------------------------------------------------------------")
    return plt

'''
Parameters:
    xTest:  (List) List of images
    yTest:  (List) List of corresponding labels

Displays data in a nice image
'''
def getAccuracyMatrix(accuracyMatrix):
    accuracyMatrixTotal = [0]*len(accuracyMatrix[0])
    for x in range(len(accuracyMatrix[0])):
        for y in range(len(accuracyMatrix)):
            accuracyMatrixTotal[x] = accuracyMatrixTotal[x]+accuracyMatrix[y][x]

    for x in range(len(accuracyMatrixTotal)):
        accuracyMatrixTotal[x] = accuracyMatrixTotal[x]/len(accuracyMatrix)

    return accuracyMatrixTotal

'''
Parameters:
    xTest:  (List) List of images
    yTest:  (List) List of corresponding labels

Displays data in a nice image
'''
def getAllAccuracyMatrix(accMatrix):
    accM = []
    for x in range(len(accMatrix[0])):
        aM = [[0]*len(accMatrix[0][0][0])for _ in range(len(accMatrix[0][0]))]
        for y in range(len(accMatrix[0][0])):
            for z in range(len(accMatrix[0][0][0])):
                sum = 0
                for i in range(len(accMatrix)):
                    sum = sum+accMatrix[i][x][y][z]
                aM[y][z] = sum
        accM.append(aM)
    for x in range(len(accM)):
        for y in range(len(accM[0])):
            for z in range(len(accM[0][0])):
                accM[x][y][z] = accM[x][y][z]/len(accMatrix)
    return accM

'''
Parameters:
    xTest:  (List) List of images
    yTest:  (List) List of corresponding labels

Displays data in a nice image
'''
def getConfusionMatrix(accMatrix):
    accM = []
    for x in range(len(accMatrix[0])):
        aM = [[0]*len(accMatrix[0][0][0])for _ in range(len(accMatrix[0][0]))]
        for y in range(len(accMatrix[0][0])):
            for z in range(len(accMatrix[0][0][0])):
                sum = 0
                for i in range(len(accMatrix)):
                    sum = sum+accMatrix[i][x][y][z]
                aM[y][z] = sum
        accM.append(aM)
    confMat = []
    for x in range(len(accM[0])):
        cM = [0]*len(accM[0][0])
        for y in range(len(accM[0][0])):
            sum = 0
            for z in range(len(accM)):
                sum = sum+accM[z][x][y]
            cM[y] = sum/len(accM)
        confMat.append(cM)
    return confMat

'''
Parameters:
    predMatrix:  A list of predictions [num of instances][num_class][num_class]
    yMatrix:     A list of corresponding correct labels [num of instances][num_class][num_class]

Returns confusion matrix
'''
def getAllConfusionMatrix (predMatrix, yMatrix):
    accMatrix = [[0]*num_classes for x in range(num_classes)]
    for x in range(len(yMatrix[0][0])):
        for z in range(len(yMatrix[0])):
            for i in range(len(yMatrix)):
                accMatrix[yMatrix[i][z][x]][predMatrix[i][z][x]] =  accMatrix[yMatrix[i][z][x]][predMatrix[i][z][x]]+1
    return accMatrix


def getClassAccuracyPlot(MT):
    figure(num=None, figsize=(26, 12), dpi=180, facecolor='w', edgecolor='k')
    for Algo in [CNN, NB, NN, KNN, SVM]:
        with open("data/Output/"+Algo+"_"+str(MT), 'rb') as f:
            [yMatrix, predMatrix, accuracyMatrix, accMatrix] = pickle.load(f)
        if Algo == KNN or Algo==SVM:
            yMatrix= [yMatrix]
            predMatrix= [predMatrix]
            accuracyMatrix= [accuracyMatrix]
            accMatrix= [accMatrix]
            accMatrix = list(np.reshape(np.array(accMatrix), (len(accMatrix),len(accMatrix[0]),10,10)))

        if MT==ShiftY or MT==ShiftX:
            for x in range (len(yMatrix)):
                yMatrix[x] = yMatrix[x][22:78]
                predMatrix[x] = predMatrix[x][22:78]
                accuracyMatrix[x] = accuracyMatrix[x][22:78]
                accMatrix[x] = accMatrix[x][22:78]

        accM = getAllAccuracyMatrix(accMatrix)
        m = len(accM)

        for j in range(num_classes):
            zero = []
            for i in range(len(accM)):
                zero.append(accM[i][j][j])
            plt.axis([-m/2, m/2, 0, 1])
            plt.subplot(2,5,j+1)
            plt.plot(np.arange(-m/2,m/2,1),zero, label=str(Algo))
            plt.xlabel("Value")
            if (MT==Rotate or MT==Shear):
                plt.xlabel("Angle")
            plt.ylabel("Accuracy")
            plt.title('Digit: '+str(j))
            plt.legend()

            sum = 0
            for x in zero:
                sum = sum+x
            sum = sum/m
            sum = sum*100
        print()
    plt.suptitle("MR: "+str(MT))
    plt.show()

def getDatasetAccuracyPlot(MT):
    figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')

    for Algo in [CNN, NB, NN, KNN, SVM]:
        with open("data/Output/"+Algo+"_"+str(MT), 'rb') as f:
            [yMatrix, predMatrix, accuracyMatrix, accMatrix] = pickle.load(f)
        if (Algo == KNN or Algo==SVM):
            yMatrix= [yMatrix]
            predMatrix= [predMatrix]
            accuracyMatrix= [accuracyMatrix]
            accMatrix= [accMatrix]
        if MT==ShiftY or MT==ShiftX:
            for x in range (len(yMatrix)):
                yMatrix[x] = yMatrix[x][22:78]
                predMatrix[x] = predMatrix[x][22:78]
                accuracyMatrix[x] = accuracyMatrix[x][22:78]
                accMatrix[x] = accMatrix[x][22:78]

        aMatrix = getAccuracyMatrix(accuracyMatrix)

        m = len(aMatrix)
        sum = 0
        for x in aMatrix:
            sum = sum+x
        sum = sum/m
        sum = sum*100
        plt.plot(np.arange(-m/2,m/2,1),aMatrix, label=str(Algo))


    plt.axis([-m/2, m/2, 0, 1])
    plt.xlabel("Value")
    if (MT==Rotate or MT==Shear):
        plt.xlabel("Angle")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.suptitle("MR: "+str(MT))
    return plt
