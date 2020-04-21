# Network Parameters
from base import *

yMatrix = []
predMatrix = []
accuracyMatrix = []
accMatrix = []

Algo = sys.argv[1]
MT = sys.argv[2]

model_dir = "models/"+Algo
#model_dir = "models/Test"+str(i)
if Algo == "CNN":
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
elif Algo == "NN":
    model = tf.estimator.Estimator(nn_model_fn, model_dir=model_dir)
yMatrixTemp = []
predMatrixTemp = []
accuracyMatrixTemp = []

for step in range(-50,50):
    with open("data/"+str(MT)+"/"+str(step+50), 'rb') as f:
        [xTest, yTest] = pickle.load(f)
    yTest = np.array(yTest)
    if Algo == NN:
        xTest = xTest.reshape(xTest.shape[0],  784)
    p,a = TestModel (model, xTest, yTest, False)
    yMatrixTemp.append(yTest)
    predMatrixTemp.append(p)
    accuracyMatrixTemp.append(a)
    print ("Step: "+str(step)+" Accuracy: "+str(a))
accMatrixTemp = processResults(predMatrixTemp, yMatrixTemp)
yMatrix.append(yMatrixTemp)
predMatrix.append(predMatrixTemp)
accuracyMatrix.append(accuracyMatrixTemp)
accMatrix.append(accMatrixTemp)
 
fname = "data/Output/"+Algo+"_"+str(MT)

if (os.path.exists(fname)):
        os.remove(fname)
with open(fname, 'wb') as f:
    pickle.dump([yMatrix, predMatrix, accuracyMatrix, accMatrix], f)



