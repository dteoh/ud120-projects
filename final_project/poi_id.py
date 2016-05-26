#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

# For data exploration
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

df = pd.DataFrame([dict(name=name, **my_dataset[name]) for name in my_dataset])

# Cleanup data
PAYMENTS = ('salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments')
STOCK_VALUE = ('exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value')

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

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
