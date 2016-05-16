#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Making the training set smaller
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# Optimizing C for RBF kernel
# for c in [10, 100, 1000, 10000]:
for c in [10000]:
    clf = SVC(C=c, kernel='rbf')
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "C={0} accuracy={1}".format(c, accuracy_score(labels_test, pred))
    # Extracting predictions
    # for elem in [10, 26, 50]:
    #     print "element={0} prediction={1}".format(elem, pred[elem])
    print "prediction=1 num={0}".format(sum(pred))

#########################################################

