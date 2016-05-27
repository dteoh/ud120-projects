#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

# For data exploration
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

df = pd.DataFrame([dict(name=name, **my_dataset[name]) for name in my_dataset])

PAYMENTS = ('salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments')
STOCK_VALUE = ('exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value')
EMAILS = ('from_messages','from_poi_to_this_person','from_this_person_to_poi','to_messages','shared_receipt_with_poi')

# Cleanup data

## Convert NaN strings into 0
df = df.replace(to_replace='NaN', value=0.)

### Identify discrepancies
df.loc[df.total_payments < (df.salary + df.bonus + df.long_term_incentive + df.deferred_income + df.deferral_payments + df.loan_advances + df.other + df.expenses + df.director_fees), ['name', 'total_payments']]
df.loc[df.total_payments > (df.salary + df.bonus + df.long_term_incentive + df.deferred_income + df.deferral_payments + df.loan_advances + df.other + df.expenses + df.director_fees), ['name', 'total_payments']]
df.loc[df.total_stock_value < (df.exercised_stock_options + df.restricted_stock + df.restricted_stock_deferred), ['name', 'total_stock_value']]
df.loc[df.total_stock_value > (df.exercised_stock_options + df.restricted_stock + df.restricted_stock_deferred), ['name', 'total_stock_value']]

### There seems to be some transposition errors in the data...
### Manually fixing bad data
df.loc[df.name == 'BELFER ROBERT', PAYMENTS] = (0. ,0., 0., -102500., 0., 0., 0., 3285., 102500., 3285.)
df.loc[df.name == 'BELFER ROBERT', STOCK_VALUE] = (0., 44093., -44093., 0.)

df.loc[df.name == 'BHATNAGAR SANJAY', PAYMENTS] = (0., 0., 0., 0., 0., 0., 0., 137864., 0., 137864.)
df.loc[df.name == 'BHATNAGAR SANJAY', STOCK_VALUE] = (15456290., 2604490., -2604490., 15456290.,)

### Drop summary row

df = df[df.name != 'TOTAL']

### Task 1: Select what features you'll use.

# Visual exploration of relationship between being a POI and individual data points.
# df.boxplot(by='poi', column=list(STOCK_VALUE))

# for row in df.itertuples():
#     plt.scatter(row.other, row.bonus, c=('r' if row.poi else None))
# plt.xlabel('other')
# plt.ylabel('bonus')
# plt.show()

# exit()

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
payment_features = list(PAYMENTS)
payment_features.remove('total_payments')
stock_value_features = list(STOCK_VALUE)
stock_value_features.remove('total_stock_value')
email_features = list(EMAILS)
features_list = ['poi']
features_list.extend(payment_features)
features_list.extend(stock_value_features)
features_list.extend(email_features)

features_list = ['poi', 'loan_advances' ,'deferral_payments' ,'exercised_stock_options' ,'long_term_incentive' ,'bonus' ,'other']

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

def convert_df_to_my_dataset():
    dataset = {}
    for row in df.itertuples():
        dataset[row.name] = row._asdict()
        dataset[row.name].pop('Index')
        dataset[row.name].pop('name')
    return dataset

my_dataset = convert_df_to_my_dataset()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.ensemble import BaggingClassifier
from sklearn.grid_search import GridSearchCV
# estimators = [('kbest'), ('svm', SVC())]
# clf = Pipeline(estimators)
parameters = dict(
        n_estimators=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        max_features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        bootstrap=[True],
        oob_score=[True, False])
clf = GridSearchCV(BaggingClassifier(random_state=42, n_jobs=-1), parameters, scoring='f1')

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

clf = clf.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
