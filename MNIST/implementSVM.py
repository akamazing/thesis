from base import *
from include import *

mnist = input_data.read_data_sets("data/Original", one_hot=False)

xTrain = mnist.train.images
yTrain = mnist.train.labels

xTrain = xTrain.reshape(xTrain.shape[0], 1, 28, 28)
xTrain = Transform (xTrain, yTrain, Rotate, 0)
xTrain = xTrain.reshape(xTrain.shape[0],  784)

def processResults2 (predMatrix, yMatrix):
    totalData = [0]*num_classes
    for y in yMatrix:
        totalData[y] = totalData[y]+1
    aMatrix = []
    accMatrix = [[0]*num_classes for x in range(num_classes)]
    for j in range(len(predMatrix)):
        accMatrix[yMatrix[j]][predMatrix[j]] += 1
            
    for x in range(len(accMatrix)):
        for y in range(len(accMatrix[x])):
            accMatrix[x][y] = round(accMatrix[x][y]/totalData[x], 3)
                
    aMatrix.append(accMatrix)
    return aMatrix

start = datetime.datetime.now()
clf = svm.SVC()
clf.fit(xTrain, yTrain)
end = datetime.datetime.now()
print(end-start)

accuracyMatrix = []
yMatrix = []
pMatrix = []
accMatrix = []

Algo = "SVM"
MT = ShiftY

for step in range(-50,50):
    accuracy = 0
    with open("data/"+str(MT)+"/"+str(step+50), 'rb') as f:
        [xTest, yTest] = pickle.load(f)

    xTest = xTest.reshape(xTest.shape[0],  784)
    
    pTemp = clf.predict(xTest)
    accuracy = accuracy_score(yTest, pTemp)
    
    accMatrixTemp = processResults2(pTemp, yTest)
    accuracyMatrix.append(accuracy)
    yMatrix.append(yTest)
    pMatrix.append(pTemp)
    accMatrix.append(accMatrixTemp)
    print("Done!")
    print("Accuracy:", accuracy)       
fname = "data/Output/"+Algo+"_"+str(MT)
if (os.path.exists(fname)):
        os.remove(fname)
with open(fname, 'wb') as f:
    pickle.dump([yMatrix, pMatrix, accuracyMatrix, accMatrix], f)