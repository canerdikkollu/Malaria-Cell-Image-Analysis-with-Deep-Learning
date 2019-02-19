import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from pyResnet.resnet import ResNet
from pyResnet import config
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os 
from keras.models import load_model
from keras import backend as K
import cv2

NUM_EPOCHS = 30
INIT_LR = 1e-1
BS = 128
totalTest = len(list(paths.list_images(config.TEST_PATH)))

valAug = ImageDataGenerator(rescale=1 / 255.0)

testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)


model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


model = load_model('best.h5')

#Evaluate on full test data 

# reset the testing generator and then use our trained model to
# make predictions on the data
#print("[INFO] evaluating network...")
#testGen.reset()
#predIdxs = model.predict_generator(testGen,
#	steps=(totalTest // BS) + 1)
#
## for each image in the testing set we need to find the index of the
## label with corresponding largest predicted probability
#predIdxs = np.argmax(predIdxs, axis=1)
#
## show a nicely formatted classification report
#print(classification_report(testGen.classes, predIdxs,
#	target_names=testGen.class_indices.keys()))



#Test for one image

im = plt.imread(os.getcwd() + "\\" + config.TEST_PATH + "\\" + testGen.filenames[0])
#resmi degistiince filename ve testgenclasses degistir
label = testGen.classes[0]
ab = cv2.resize(im, (64,64))
ab = np.expand_dims(ab, axis=0)
Idxs = model.predict(ab)
predict = np.argmax(Idxs, axis=1)

plt.imshow(im)

print("Predicted: " + str(predict[0]) + " Label: " + str(label))

if predict == label:
    if predict == 0:
        print("Parasitized")
    else:
        print("Uninfected")
else:
    print("False Prediction")
