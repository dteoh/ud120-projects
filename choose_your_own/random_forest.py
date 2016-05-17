#!/usr/bin/python

from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


for n_estimators in [2, 4, 8, 16, 32, 64, 128]:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    print "training_time={0}s accuracy={1} n_estimators={2}".format(round(t1 - t0, 3), accuracy_score(labels_test, pred), n_estimators)
    prettyPicture(clf, features_test, labels_test, "n_estimators{0}".format(n_estimators))

for min_samples_split in [2, 4, 8, 16, 32, 64, 128]:
    clf = RandomForestClassifier(min_samples_split=min_samples_split)
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    print "training_time={0}s accuracy={1} min_samples_split={2}".format(round(t1 - t0, 3), accuracy_score(labels_test, pred), min_samples_split)
    prettyPicture(clf, features_test, labels_test, "min_samples_split_{0}".format(min_samples_split))

for max_depth in [2, 4, 8, 16, 32, 64, 128]:
    clf = RandomForestClassifier(max_depth=max_depth)
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    print "training_time={0}s accuracy={1} max_depth={2}".format(round(t1 - t0, 3), accuracy_score(labels_test, pred), max_depth)
    prettyPicture(clf, features_test, labels_test, "max_depth_{0}".format(max_depth))

for max_features in ['sqrt', 'log2', None]:
    clf = RandomForestClassifier(max_features=max_features)
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    print "training_time={0}s accuracy={1} max_features={2}".format(round(t1 - t0, 3), accuracy_score(labels_test, pred), max_features)
    prettyPicture(clf, features_test, labels_test, "max_features_{0}".format(max_features))

# Hand tuned from above results
clf = RandomForestClassifier(n_estimators=32, min_samples_split=4, max_depth=64)
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
pred = clf.predict(features_test)
print "training_time={0}s accuracy={1}".format(round(t1 - t0, 3), accuracy_score(labels_test, pred), min_samples_split)
prettyPicture(clf, features_test, labels_test, 'final')
