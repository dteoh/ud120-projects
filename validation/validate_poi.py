#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
print clf.score(features, labels)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)

print "expected={0}".format(len([y for y in labels_test if y > 0]))

predictions = clf.predict(features_test)
print "predicted={0}".format(len([y for y in predictions if y > 0]))
print "true_positives={0}".format(len([1 for (expected, predicted) in zip(labels_test, predictions) if expected == predicted and expected == 1]))

from sklearn.metrics import precision_score, recall_score
print "precision={0}".format(precision_score(labels_test, predictions))
print "recall={0}".format(recall_score(labels_test, predictions))


# [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
# [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# TP = 6
# FP = 3
# FN = 2

