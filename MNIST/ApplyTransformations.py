# Base code include
from include import *
from base import *

datasetLocation = "data/Original" # Directory containing the dataset
MT = Rotate       # Metamorphic property
startValue = -50  # starting value of transformation
endValue = 50     # last value of transformation
step = 1          # step
num_classes = 10  # Number of classes
num_images = 1000 # Number of images to pick from each class
outputDir = "data/+str(MT)/"   # output directory


# Read dataset
mnist = input_data.read_data_sets(datasetLocation, one_hot=False)

# Load data to be tansformed
xTest = mnist.test.images
yTest = mnist.test.labels

# Reshape it as image
xTest = xTest.reshape(xTest.shape[0], 1, 28, 28)

# Temporary variables
yT = []
xT = []

# Create a set of num_images images from each class
for n in range(num_classes):
    train_filter = np.isin(yTest, [n])
    x, y = xTest[train_filter], yTest[train_filter]
    yT.append(list(y[:num_images]))
    xT.append(x[:num_images])
yTest = [i for l in yT for i in l]
xT = np.array(xT)
xTest = np.concatenate(xT)

# Start the transformation
for x in range(startValue,endValue,step):
    print("MR: "+str(MT)+", Step: "+str(x))
    fout = "outputDir"+str(x)
    # Remove output directory if exists
    if (os.path.exists(fout)):
        os.remove(fout)
    # Transform the data
    x_test = Transform (xTest, yTest, MT, x)
    # Save the data
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)
