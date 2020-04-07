#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pickle
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'pylab inline')
# Change figure size into 8 by 6 inches
matplotlib.rcParams['figure.figsize'] = (8, 6)

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing


# In[15]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']
features_list = poi_label + email_features + financial_features


# In[4]:


# Load the dataset
with open('final_project_dataset.pkl', 'rb') as data_file:
    data_dict = pickle.load(data_file)


# In[16]:


# Total number of data points
print("Total number of data points: %i" %len(data_dict))
# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (len(data_dict) - poi))
       


# In[7]:


# Number of features
print('There are {} features.'.format(len(list(data_dict.values())[0])))
# Names of features
list(list(data_dict.values())[0].keys())


# In[8]:


# Features with missing values
nan_counts_poi = defaultdict(int)
nan_counts_non_poi = defaultdict(int)
for data_point in data_dict.values():
    if data_point['poi'] == True:
        for feature, value in data_point.items():
            if value == "NaN":
                nan_counts_poi[feature] += 1
    elif data_point['poi'] == False:
        for feature, value in data_point.items():
            if value == "NaN":
                nan_counts_non_poi[feature] += 1
    else:
        print('Got an uncategorized person.')
nan_counts_df = pd.DataFrame([nan_counts_poi, nan_counts_non_poi]).T
nan_counts_df = nan_counts_df.fillna(value=0)
nan_counts_df.columns = ['# NaN in POIs', '# NaN in non-POIs']
nan_counts_df['# NaN total'] = nan_counts_df['# NaN in POIs'] +                                nan_counts_df['# NaN in non-POIs']
nan_counts_df['% NaN in POIs'] = nan_counts_df['# NaN in POIs'] /                                           poi_counts[True] * 100
nan_counts_df['% NaN in non-POIs'] = nan_counts_df['# NaN in non-POIs'] /                                           poi_counts[False] * 100
nan_counts_df['% NaN total'] = nan_counts_df['# NaN total'] /                                           len(data_dict) * 100
    
nan_counts_df


# In[9]:


# Plot missing values distribution
ax = nan_counts_df.sort_values('# NaN total')[['% NaN in POIs', '% NaN in non-POIs']].plot(kind='barh', 
                                                                                    stacked=False)
ax.set_title('Percent of Missing Values Distribution')
ax.set_xlabel('Percent of Missing Values')
ax.set_ylabel('Variable Name')


# In[17]:


### Task 2: Remove outliers
def plotOutliers(data_set, feature_x, feature_y):
    """
    This function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()
# Visualize data to identify outliers
print(plotOutliers(data_dict, 'total_payments', 'total_stock_value'))
print(plotOutliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(plotOutliers(data_dict, 'salary', 'bonus'))
print(plotOutliers(data_dict, 'total_payments', 'other'))
identity = []
for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        identity.append((person, data_dict[person]['total_payments']))
print("Outlier:")
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

# Find persons whose financial features are all "NaN"
fi_nan_dict = {}
for person in data_dict:
    fi_nan_dict[person] = 0
    for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            fi_nan_dict[person] += 1
sorted(fi_nan_dict.items(), key=lambda x: x[1])

# Find persons whose email features are all "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])


# In[18]:


# Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LAY KENNETH L", 0)
data_dict.pop("FREVERT MARK A", 0)
data_dict.pop("BHATNAGAR SANJAY", 0)


# In[24]:


### Task 3: Create new feature(s)
### Lets Create The Follwoing New Features
### from_this_person_to_poi_ratio
### from_poi_to_this_person_ratio
### salary_bonus_ratio

def calculate_ratios(val1, val2):
    result = 0
    if val1 == 'NaN' or val2 == 'NaN':
        result = 'NaN'
    else :
        result = val1/float(val2)

    return result

for key,value in data_dict.items():
    value['from_this_person_to_poi_ratio'] = calculate_ratios(value['from_this_person_to_poi'], value['from_messages'])
    value['from_poi_to_this_person_ratio'] = calculate_ratios(value['from_poi_to_this_person'], value['to_messages'])
    value['bonus_salary_ratio'] = calculate_ratios(value['bonus'], value['salary'])


# In[62]:



### Store to my_dataset for easy export below.
my_dataset = data_dict

###The below list of features were chosen by applying SelectKbest process
###The Top 11 features were chosen from the list for further analysis
###The selectKBest code which is below lists the importance of each feature
features_list = [
 'poi',
 'exercised_stock_options',
 'total_stock_value',
 'bonus',
 'salary',
 'from_this_person_to_poi_ratio',
 'deferred_income',
 'bonus_salary_ratio',
 'long_term_incentive',
 'restricted_stock',
#'total_payments',
#'shared_receipt_with_poi',
#'loan_advances',
#'expenses',
#'from_poi_to_this_person',
#'other',
#'from_poi_to_this_person_ratio',
#'from_this_person_to_poi',
#'director_fees',
#'to_messages',
#'deferral_payments',
#'from_messages',
#'restricted_stock_deferred'
]


# In[63]:


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print 'Total Number Of Data Points After Removing Outliers And Feature Formatting : ', len(features)

#Code to print importance of each feature and to select best out of them for further analysis
'''selector = SelectKBest(f_classif, k='all')
selector.fit(features, labels)
feature_importances_skb = selector.scores_
features_list_with_imp = []
for i in range(len(feature_importances_skb)):
    features_list_with_imp.append([features_list[i+1], feature_importances_skb[i]])
features_list_with_imp = sorted(features_list_with_imp, reverse=True, key=itemgetter(1))
for i in range(len(features_list_with_imp)):
    print features_list_with_imp[i]'''


# In[64]:



#Scalling The Data Using MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#Splitting Data Into Test And Train Data
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[65]:


for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0
new_features_list = features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio']

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Select the best features: 
#Removes all features whose variance is below 80% 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features
from sklearn.feature_selection import f_classif
k = 7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list +                      ['msg_to_poi_ratio', 'msg_from_poi_ratio'],                      sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data)
new_f_features = scaler.fit_transform(new_f_features)


# In[53]:


## Task 4: Try a varity of classifiers
## Please name your classifier clf for easy export below.
## Note that if you want to do PCA or other multi-stage operations,
## you'll need to use Pipelines. For more info:
## http://scikit-learn.org/stable/modules/pipeline.html

def evaluate_clf(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


# In[42]:


from sklearn import naive_bayes        
nb_clf = naive_bayes.GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(nb_clf, nb_param)

print("Evaluate naive bayes model")
evaluate_clf(nb_grid_search, features, labels, nb_param)


# In[47]:


print("Evaluate naive bayes model using dataset with new features")
evaluate_clf(nb_grid_search, new_f_features, new_f_labels, nb_param)


# In[49]:


from sklearn import linear_model
from sklearn.pipeline import Pipeline
lo_clf = Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler()),
        ('classifier', linear_model.LogisticRegression())])
         
lo_param = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001],             'classifier__C': [0.1, 0.01, 0.001, 0.0001]}
lo_grid_search = GridSearchCV(lo_clf, lo_param)
print("Evaluate logistic regression model")
evaluate_clf(lo_grid_search, features, labels, lo_param)


# In[50]:


from sklearn import svm
s_clf = svm.SVC()
s_param = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}    
s_grid_search = GridSearchCV(s_clf, s_param)
print("Evaluate svm model")
evaluate_clf(s_grid_search, features, labels, s_param)


# In[56]:


# Provided to give you a starting point. Try a variety of classifiers.
## Gaussian Naive Bayes Classifier
clf = GaussianNB()

### Decision Tree Classifier
## Best Params Reported By GridSearchCV {'min_samples_split': 11, 'splitter': 'random', 'criterion': 'entropy', 'max_depth': 2, 'class_weight': None}
clf = DecisionTreeClassifier(min_samples_split=11, splitter='random', criterion='entropy', max_depth=2, class_weight=None)
clf = DecisionTreeClassifier()

### Random Forest Classifier
## Best Params Reported By GridSearchCV {'min_samples_split': 4, 'criterion': 'gini', 'max_depth': 7, 'class_weight': None}
clf = RandomForestClassifier(min_samples_split=4, criterion='gini', max_depth=7, class_weight=None)

## KNeighborsClassifier
## Best Params Reported By GridSearchCV  {'n_neighbors': 6, 'weights': 'uniform', 'algorithm': 'auto'}
clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='auto')

## Params For Tuning DecisionTree
'''params = {'criterion':['gini','entropy'],
          'max_depth':[i for i in range(2,15)],
          'min_samples_split':[i for i in range(2,15)],
          #'splitter' : ['best', 'random'],
          'class_weight' : [None, 'balanced']
         }'''

## Params For Tuning KNN
'''params = {'n_neighbors' : [i for i in range(5,20)],
          'weights': ['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
          }'''



clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


# In[57]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print '------------------------------------------------------'
print 'Metrics:'
print 'Recall Score: ', recall_score(labels_test, pred)
print 'Precision Score: ', precision_score(labels_test, pred)
print classification_report(labels_test, pred)
print '------------------------------------------------------'


# In[ ]:




