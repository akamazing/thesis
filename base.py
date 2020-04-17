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


# In[ ]:


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


# In[ ]:


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
        


# In[ ]:
def TestModel(model, xData, yData, shuffle):
    #input_fn = tf.estimator.inputs.numpy_input_fn(
    #        x={'images': xData}, y= np.array(yData),
    #        batch_size=batch_size, shuffle=shuffle)
    #accuracy = model.evaluate(input_fn)

    # Prepare the input data
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': xData}, shuffle=shuffle)
    # Use the model to predict the images class
    preds = list(model.predict(input_fn))
    acc = 0.0
    for x in range(len(yData)):
        if (yData[x]==preds[x]):
            acc = acc+1
    acc = acc/len(yData)
    #return preds, accuracy['accuracy']
    return preds, acc


# In[ ]:


def TrainModel(model, xTrain, yTrain, shuffle):
    input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': xTrain}, y= np.array(yTrain),
            batch_size=batch_size, num_epochs=None, shuffle=shuffle)
    # Train the Model
    model.train(input_fn, steps=num_steps)


# In[ ]:


def Transform( xData, yData, tType, value ): # yData is not bein transformed
    
    n_images = len(xData)
    xTemp = []
    yTemp = []
    
    datagen = ImageDataGenerator()# fit parameters from data
    
    # tType = 1 for shade
    if tType == "Shade":
        Xnew = [[[[v-value if v-value>0.0 else 0.0 for v in n] for n in x[0]]] for x in xData]
        # Xnew = [[[[v-value for v in n] for n in x[0]]] for x in xData]
        Xnew = np.array(Xnew)
        xTemp[:] = Xnew.astype('float32')

    # tType = 2 for rotation
    if tType == "Rotate":
        #datagen = ImageDataGenerator(rotation_range=value)# fit parameters from data
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'theta':value})
            xTemp.append(x)
            
    # tType = 3 for sheer
    if tType == "Shear":
        #datagen = ImageDataGenerator(shear_range=value)# fit parameters from data
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'shear':value})
            xTemp.append(x)
        
    # tType = 3 for shift
    if tType == "ShiftX":   
        #datagen = ImageDataGenerator(width_shift_range=value, height_shift_range=value)
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'tx':value})
            xTemp.append(x)
            
    # tType = 3 for shift
    if tType == "ShiftY":   
        #datagen = ImageDataGenerator(width_shift_range=value, height_shift_range=value)
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'ty':value})
            xTemp.append(x)
            
    # tType = 4 for shift
    if tType == "ZoomX":   
        #datagen = ImageDataGenerator(width_shift_range=value, height_shift_range=value)
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'zx':value})
            xTemp.append(x)
            
     # tType = 5 for shift
    if tType == "ZoomY":   
        #datagen = ImageDataGenerator(width_shift_range=value, height_shift_range=value)
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'zy':value})
            xTemp.append(x)
        
    #datagen.fit(xData)
    # configure batch size and retrieve one batch of images
    #for xBatch, yBatch in datagen.flow(xData, yData, batch_size=n_images, shuffle=False):
        #xData[:] = xBatch
        #yData[:] = yBatch
        #break
    xTmp = []
    for x in xTemp:
        xTmp.append(np.transpose(x))
    xTmp = np.array(xTmp)
    return xTmp


# In[ ]:


def DisplayData( xTest, yTest ):
    for i in range(len(xTest)):
        plt.imshow(np.reshape(xTest[i], [28, 28]), cmap='gray')
        plt.show()
        print ("Image Label: ", yTest[i])


# In[ ]:


def OpenFile(MT):
    filename = str(MT)+".txt"
    fp = open(filename, 'w')
    return fp

def CloseFile(fp):
    fp.close()
    
#Shade = "Shade"
#Rotate = "Rotate"
#Sheer = "Sheer"
#Shift = "Shift"


# In[ ]:


def ShuffleMT (mode, fp, iteration):
    print ("Reloading training and test data")
    reloadData(Train)
    reloadData(Test)
    
    if (mode==Train):
        print ("Training model")
        TrainModel(xTrain, yTrain, True)
        
        print ("Evaluating Model")
        accuracy = EvaluateModel(xTest, yTest, False)
          
    elif (mode==Test):
        print ("Evaluating Model")
        accuracy = EvaluateModel(xTest, yTest, True)
        
    elif (mode=="Both"):
        print ("Training model")
        TrainModel(xTrain, yTrain, True)       
        print ("Evaluating Model")
        accuracy = EvaluateModel(xTest, yTest, True)

    print("Iteration: ", iteration, " Accuracy: ", accuracy)
    if (fp):
        fp.write(str(iteration)+"\t"+str(accuracy)+"\n")


# In[ ]:


def processResults (predMatrix, yMatrix):
    totalData = [0]*num_classes
    for y in yMatrix[0]:
        totalData[y] = totalData[y]+1
    aMatrix = []
    for i in range(len(predMatrix)):       
        accMatrix = [[0]*num_classes for x in range(num_classes)]
        for j in range(len(predMatrix[i])):
            accMatrix[yMatrix[i][j]][predMatrix[i][j]] += 1
            
        for x in range(len(accMatrix)):
            for y in range(len(accMatrix[x])):
                accMatrix[x][y] = round(accMatrix[x][y]/totalData[x], 3)
                
        aMatrix.append(accMatrix)
    return aMatrix


# In[ ]:
def getAccuracyPlot(MT, accMatrix):
    figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    m = len(accMatrix)
    plt.axis([-m/2, m/2, 0, 1])
    plt.plot(np.arange(-m/2,m/2,1),accMatrix)
    plt.xlabel("Value")
    if (MT==Rotate or MT==Shear):
        plt.xlabel("Angle")
    plt.ylabel("Accuracy")
    #plt.suptitle("MR: "+str(MT))
    return plt
    
def getAllAccuracyPlot(MT, accMatrix):
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
        plt.ylabel("Accuracy")
        plt.title('Digit: '+str(j))
    #plt.suptitle("MR: "+str(MT))
    plt.show()
    return plt

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

def getAccuracyMatrix(accuracyMatrix):
    accuracyMatrixTotal = [0]*len(accuracyMatrix[0])
    for x in range(len(accuracyMatrix[0])):
        for y in range(len(accuracyMatrix)):
            accuracyMatrixTotal[x] = accuracyMatrixTotal[x]+accuracyMatrix[y][x]

    for x in range(len(accuracyMatrixTotal)):
        accuracyMatrixTotal[x] = accuracyMatrixTotal[x]/len(accuracyMatrix)
    
    return accuracyMatrixTotal

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

def getAllConfusionMatrix (predMatrix, yMatrix):
    accMatrix = [[0]*num_classes for x in range(num_classes)] 
    for x in range(len(yMatrix[0][0])):
        for z in range(len(yMatrix[0])):
            for i in range(len(yMatrix)):
                accMatrix[yMatrix[i][z][x]][predMatrix[i][z][x]] =  accMatrix[yMatrix[i][z][x]][predMatrix[i][z][x]]+1
    return accMatrix

def getAccuracyMatrix(accuracyMatrix):
    accuracyMatrixTotal = [0]*len(accuracyMatrix[0])
    for x in range(len(accuracyMatrix[0])):
        for y in range(len(accuracyMatrix)):
            accuracyMatrixTotal[x] = accuracyMatrixTotal[x]+accuracyMatrix[y][x]

    for x in range(len(accuracyMatrixTotal)):
        accuracyMatrixTotal[x] = accuracyMatrixTotal[x]/len(accuracyMatrix)
    
    return accuracyMatrixTotal

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
            cM[y] = sum
        confMat.append(cM)
    return confMat

# the same as scikit.learn. confusion_matrix
def getAllConfusionMatrix (predMatrix, yMatrix):
    accMatrix = [[0]*num_classes for x in range(num_classes)] 
    for x in range(len(yMatrix[0][0])):
        for z in range(len(yMatrix[0])):
            for i in range(len(yMatrix)):
                accMatrix[yMatrix[i][z][x]][predMatrix[i][z][x]] =  accMatrix[yMatrix[i][z][x]][predMatrix[i][z][x]]+1
    return accMatrix

