#James Alford-Golojuch
#First attempt at using an MLP Classifier for classification of images

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#Initializes training data and labels and testing data from given data files
data_train = np.genfromtxt('caltechTrainData.dat')
data_train = data_train / 255
data_labels = np.genfromtxt('caltechTrainLabel.dat')
data_test = np.genfromtxt('caltechTestData.dat')
data_test = data_test / 255

#Initializes the cneural network classifier settings
clf = MLPClassifier(activation='relu', alpha=1e-06, batch_size='auto', 
       early_stopping=False, epsilon=1e-08, learning_rate='constant',
       learning_rate_init=0.001, max_iter=1500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
clf.fit(data_train,data_labels)
results = clf.score(data_train, data_labels)
pred = clf.predict(data_train)
print(results)
print(pred)

#Confusion matrix for training data
confMatrix = confusion_matrix(data_labels, pred)
print(confMatrix)

testLabels = clf.predict(data_test)
print(testLabels)

f = open('caltechPredictLabel.dat', 'w')
for x in range(0,len(testLabels)):
    f.write(str(testLabels[x]))
    f.write('\n')
    
f.close()