# Training Parameters
from base import *

# Training Parameters
mnist = input_data.read_data_sets("data/Original", one_hot=False)

# Load the Mnist test data
xTmp = mnist.test.images
yTmp = list(mnist.test.labels-1)
xTmp = xTmp.reshape(xTmp.shape[0], 1, 28, 28)
# # convert from int to float
xTmp = xTmp.astype('float32')
xTest = []
for x in xTmp:
    xTest.append(np.transpose(x))
xTest = np.array(xTest)
xTest = xTest.reshape(xTest.shape[0],  784)
yTest = yTmp
xTestBackup = xTest.copy()
yTestBackup = yTmp.copy()

# Load the Mnist test data
xTmp = mnist.train.images
yTmp = list(mnist.train.labels-1)
xTmp = xTmp.reshape(xTmp.shape[0], 1, 28, 28)
# # convert from int to float
xTmp = xTmp.astype('float32')
xTrain = []
for x in xTmp:
    xTrain.append(np.transpose(x))
xTrain = np.array(xTrain)
xTrain = xTrain.reshape(xTrain.shape[0],  784)
yTrain = yTmp
xTrainBackup = xTrain.copy()
yTrainBackup = yTmp.copy()

Algo = CNN

   
model_dir = "models/"+Algo
#model_dir = "models/Test"+str(i)
if (os.path.exists(model_dir)):
    shutil.rmtree(model_dir)
if Algo == "CNN":
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
elif Algo == "NN":
    model = tf.estimator.Estimator(nn_model_fn, model_dir=model_dir)

TrainModel (model, xTrain, yTrain, False)
p,a = TestModel (model, xTest, yTest, False)
print ("Step: "+" Accuracy: "+str(a))    
