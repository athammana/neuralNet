from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#Abhijay Thammana

#I know you don't need semicolons but I am really used to typing with them sorry

#numpy arrays
images = np.load('images.npy');
labels = np.load('labels.npy');

#reshapes label to make an array with a 1 at the label index
newLabel = [[0 for x in range(10)] for y in range(len(labels))];
for label in range(len(labels) - 1):
	arr = np.zeros((10,), dtype=int);
	arr[labels[label]] = 1;
	newLabel[label] = (arr);
labels = np.array(newLabel);
#reshape the image array from 6500x28x28 to 6500x784
newImages = np.array([[0 for x in range(784)] for y in range(len(labels))])
for x in range(len(images)):
	newImages[x] = images[x].flatten();
images = newImages;

trainImages = images[int((.4*len(images))):];
trainLabels = labels[int((.4*len(labels))):];

valImages = images[int((.2*len(images))):int((.4*len(images)))];
valLabels = labels[int((.2*len(labels))):int((.4*len(labels)))];

testImages = images[:int((.2*len(images)))];
testLabels = labels[:int((.2*len(labels)))];

trainImages = images[:int(1*len(images)/3)] + images[int(2*len(images)/3)+1:];
trainLabels = labels[:int(1*len(labels)/3)] + labels[int(2*len(labels)/3)+1:];

print(trainLabels);

#creating actual ann model
def setModel():
	model = Sequential();
	#Dropout layer
	model.add(Dropout(.2, input_shape = (784, )));
	model.add(Dense(50, activation = 'relu', input_dim = 784));
	for i in range(10):
		model.add(Dense(50, activation = 'relu'));
	model.add(Dense(10, activation = 'softmax'));
	# Created 10 hidden layers of 50 nodes an input layer of 784 nodes and output of 10

	sgd = optimizers.SGD(lr = 0.001);

	model.compile(optimizer = sgd,#'rmsprop', 
				  loss = 'categorical_crossentropy', 
			  metrics = ['accuracy']);
	return model;

def tryNormalModel(model):
	history = model.fit(trainImages, trainLabels, epochs = 100, batch_size = 32, validation_data = (valImages, valLabels));
	score = model.evaluate(testImages, testLabels, batch_size = 32);
	print("Score: ", score);
	printConfusion(history);

def kCrossVal(model, images, labels):
	scores = [];
	for k in range(3):
		if(k == 0):
			# blocks of data: [train] | [train] | [validation]
			trainImages = images[:int(2*len(images)/3)];
			trainLabels = labels[:int(2*len(labels)/3)];
			valImages = images[int(2*len(images)/3):];
			valLabels = labels[int(2*len(labels)/3):];
		elif(k == 1):
			# blocks of data: [train] | [validation] | [train]
			trainImages = images[:int(1*len(images)/3)] + images[int(2*len(images)/3)+1:];
			trainLabels = labels[:int(1*len(labels)/3)] + labels[int(2*len(labels)/3)+1:];
			valImages = images[int(1*len(images)/3):int(2*len(images)/3)];
			valLabels = labels[int(1*len(labels)/3):int(2*len(labels)/3)];
		elif(k == 2):
			# blocks of data: [validation] | [train] | [train]
			trainImages = images[int(1*len(images)/3):];
			trainLabels = labels[int(1*len(labels)/3):];
			valImages = images[:int(1*len(images)/3)];
			valLabels = labels[:int(1*len(labels)/3)];

		history = model.fit(trainImages, trainLabels, epochs = 100, batch_size = 32, validation_data = (valImages, valLabels));

		score = model.evaluate(testImages, testLabels, batch_size = 32);
		# print("Cross, ", k, "Score: ", score);
		scores.append(score);
		# printConfusion(model);
		# plotHis(history);
	print(scores);


def printConfusion(model):
	#confusion matrix
	testPred = model.predict(testImages);
	matrix = confusion_matrix(testLabels.argmax(axis = 1), testPred.argmax(axis = 1));
	print("COnfusion, ", matrix);

# plot(history);

def plotHis(history):
	print(history.history.keys());
	plt.plot(history.history['acc']);
	plt.plot(history.history['val_acc']);
	plt.title('model acc');
	plt.ylabel('accuracy');
	plt.xlabel('epoch');
	plt.legend(['train', 'test'], loc = 'upper left');
	plt.show();

	plt.plot(history.history['loss']);
	plt.plot(history.history['val_loss']);
	plt.title('model loss');
	plt.ylabel('loss');
	plt.xlabel('epoch');
	plt.legend(['train', 'test'], loc = 'upper left');
	plt.show();

model = setModel();
model.summary();
# Uncomment Function below to get Task 2
# tryNormalModel(model);
# Uncomment Function below to get Task 3
kCrossVal(model, images, labels);
# Uncomment Function below to get Task 4
# hyperParameterTesting(model);