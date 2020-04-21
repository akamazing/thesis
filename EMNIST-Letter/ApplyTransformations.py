# Training Parameters
from include import *
from base import *


# Training Parameters
mnist = input_data.read_data_sets("data/Original", one_hot=False)

xTest = mnist.test.images
yTest = mnist.test.labels

xTest = xTest.reshape(xTest.shape[0], 1, 28, 28)

#reloadData(Test)
yT = []
xT = []
for n in range(num_classes):
    train_filter = np.isin(yTest, [n+1])
    x, y = xTest[train_filter], yTest[train_filter]
    yT.append(list(y[:num_images]))
    xT.append(x[:num_images])
yTest = [i-1 for l in yT for i in l]
xT = np.array(xT)
xTest = np.concatenate(xT)

MT = sys.argv[1]

# for step in range(-50,50):
#     print("MR: "+str(MT)+", Step: "+str(step+50))
#     fout = "data/"+str(MT)+"/"+str(step+50)
#     if (os.path.exists(fout)):
#         os.remove(fout)
#     value = step*0.5
#     x_test = Transform (xTest, yTest, MT, value)
#     with open(fout, 'wb') as f:
#         pickle.dump([x_test, yTest], f)
# for step in range(-50,50):
#     print("MR: Rotate, Step: "+str(step+50))
#     fout = "variables/Transformations/Rotate/"+str(step+50)
#     if (os.path.exists(fout)):
#         os.remove(fout)
#     x_test = Transform (xTest, yTest, Rotate, step)
#     with open(fout, 'wb') as f:
#         pickle.dump([x_test, yTest], f)


# for step in range(-50,50):
#     print("MR: Shear, Step: "+str(step+50))
#     fout = "variables/Transformations/Shear/"+str(step+50)
#     if (os.path.exists(fout)):
#         os.remove(fout)
#     x_test = Transform (xTest, yTest, Shear, step)
#     with open(fout, 'wb') as f:
#         pickle.dump([x_test, yTest], f)
        
for step in range(-50,50):
    print("MR: ShiftX, Step: "+str(step+50))
    fout = "data/ShiftX/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    x_test = Transform (xTest, yTest, ShiftX, step)
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)


# for step in range(-50,50):
#     print("MR: ShiftY, Step: "+str(step+50))
#     fout = "data/ShiftY/"+str(step+50)
#     if (os.path.exists(fout)):
#         os.remove(fout)
#     x_test = Transform (xTest, yTest, ShiftY, step)
#     with open(fout, 'wb') as f:
#         pickle.dump([x_test, yTest], f)
        
# for step in range(0,100):
#     print("MR: Rotate, Step: "+str(step))
#     fout = "data/"+str(Shade)+"/"+str(step)
#     value = step*0.01
#     if (os.path.exists(fout)):
#         os.remove(fout)
#     x_test = Transform (xTest, yTest, Shade, value)
#     with open(fout, 'wb') as f:
#         pickle.dump([x_test, yTest], f)