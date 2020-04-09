#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle
import sys
import numpy as np
import pandas as pd
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# In[9]:


# Converting the given pickled Enron data to a pandas dataframe.
enron_df = pd.DataFrame.from_records(list(data_dict.values()))

# Set the index of df to be the employees series:
employees = pd.Series(list(data_dict.keys()))
enron_df.set_index(employees, inplace=True)
enron_df.head()


# In[10]:


print "Size of the enron dataframe: ", enron_df.shape
print "Number of data points (people) in the dataset: ", len(enron_df)
print "Number of Features in the Enron Dataset: ", len(enron_df.columns)

# Counting the number of POIs and non-POIs in the given dataset.
poi_count = enron_df.groupby('poi').size()
print "Total number of POI's in the given dataset: ", poi_count.iloc[1]
print "Total number of non-POI's in the given dataset: ", poi_count.iloc[0]


# In[11]:


enron_df.dtypes


# In[12]:


# Converting the datatypes in the given pandas dataframe 
# into floating points for analysis and replace NaN with zeros.

# Coerce numeric values into floats or ints; also change NaN to zero.
enron_df_new = enron_df.apply(lambda x : pd.to_numeric(x, errors = 'coerce')).copy().fillna(np.nan)
enron_df_new.head()


# In[13]:


# Dropping column 'email_address' as it is not required in analysis.
enron_df_new.drop('email_address', axis = 1, inplace = True)

# Checking the changed shape of df.
enron_df_new.shape


# ## Outlier Investigation & Analyzing the Features
# The features can be categorized as the following.
# 
# ### Financial Features (in US dollars):
# salary deferral_payments total_payments loan_advances bonus restricted_stock_deferred deferred_income total_stock_value expenses exercised_stock_options other long_term_incentive restricted_stock director_fees
# 
# ### Email Features (count of emails):
# to_messages email_address from_poi_to_this_person from_messages from_this_person_to_poi shared_receipt_with_poi
# 
# ### POI Labels (boolean):
# poi
# 
# ### Financial Features: Bonus and Salary
# 
# Drawing scatterplot of Bonus vs Salary of Enron employees.

# In[14]:


plt.scatter(enron_df_new['salary'][enron_df_new['poi'] == True],
            enron_df_new['bonus'][enron_df_new['poi'] == True], 
            color = 'r', label = 'POI')

plt.scatter(enron_df_new['salary'][enron_df_new['poi'] == False],
            enron_df_new['bonus'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')
    
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.title("Scatterplot of Salary vs Bonus w.r.t POI")
plt.legend(loc='upper left')
plt.show() 


# In[15]:


# Finding the non-POI employee having maximum salary
enron_df_new['salary'].argmax()


# In[16]:


# Deleting the row 'Total' from the dataframe
enron_df_new.drop('TOTAL', axis = 0, inplace = True)

# Drawing scatterplot with the modified dataframe
plt.scatter(enron_df_new['salary'][enron_df_new['poi'] == True],
            enron_df_new['bonus'][enron_df_new['poi'] == True], 
            color = 'r', label = 'POI')

plt.scatter(enron_df_new['salary'][enron_df_new['poi'] == False],
            enron_df_new['bonus'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')
    
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.title("Scatterplot of Salary vs Bonus w.r.t POI")
plt.legend(loc='upper left')
plt.show() 


# In[17]:


enron_df_new['bonus-to-salary_ratio'] = enron_df_new['bonus']/enron_df_new['salary']


# In[18]:


# Features of the index 'THE TRAVEL AGENCY IN THE PARK'
enron_df_new.loc['THE TRAVEL AGENCY IN THE PARK']


# In[19]:


# Deleting the row with index 'THE TRAVEL AGENCY IN THE PARK'
enron_df_new.drop('THE TRAVEL AGENCY IN THE PARK', axis = 0, inplace = True)


# In[20]:


enron_df_new['deferred_income'].describe()


# In[21]:


# Finding out the integer index locations of POIs and non-POIs.
poi_rs = []
non_poi_rs = []
for i in range(len(enron_df_new['poi'])):
    if enron_df_new['poi'][i] == True:
        poi_rs.append(i+1)
    else:
        non_poi_rs.append(i+1)

print("Length of po list: ", len(poi_rs))
print("Length non-poi list: ", len(non_poi_rs))


# In[22]:


# Since 'deferred_income' is negative, for intuitive understanding,
# a positive person of the variable is created for visualization.
enron_df_new['deferred_income_p'] = enron_df_new['deferred_income'] * -1

plt.scatter(non_poi_rs,
            enron_df_new['deferred_income_p'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(poi_rs,
            enron_df_new['deferred_income_p'][enron_df_new['poi'] == True],
            color = 'r', label = 'POI')
    
plt.xlabel('Employees')
plt.ylabel('deferred_income')
plt.title("Scatterplot of Employees with deferred income")
plt.legend(loc='upper right')
plt.show()


# In[23]:


# Scatterplot of total_payments vs deferral_payments w.r.t POI
plt.scatter(enron_df_new['total_payments'][enron_df_new['poi'] == False],
            enron_df_new['deferral_payments'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(enron_df_new['total_payments'][enron_df_new['poi'] == True],
            enron_df_new['deferral_payments'][enron_df_new['poi'] == True],
            color = 'r', label = 'POI')

plt.xlabel('Total_payments')
plt.ylabel('deferral_payments')
plt.title("Scatterplot of total_payments vs deferral_payments w.r.t POI")
plt.legend(loc='upper right')
plt.show() 


# In[24]:


# Finding the non-POI employee having maximum 'deferral_payments'
enron_df_new['deferral_payments'].argmax()


# In[25]:


# Removing the non-POI employee having maximum 'deferral_payments'
enron_df_new.drop('FREVERT MARK A', axis = 0, inplace = True)


# In[26]:


# Finding out the integer index locations of POIs and non-POIs
poi_rs = []
non_poi_rs = []
for i in range(len(enron_df_new['poi'])):
    if enron_df_new['poi'][i] == True:
        poi_rs.append(i+1)
    else:
        non_poi_rs.append(i+1)

# Making a scatterplot
plt.scatter(non_poi_rs,
            enron_df_new['long_term_incentive'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(poi_rs,
            enron_df_new['long_term_incentive'][enron_df_new['poi'] == True],
            color = 'r', label = 'POI')

plt.xlabel('Employees')
plt.ylabel('long_term_incentive')
plt.title("Scatterplot of Employee Number with long_term_incentive")
plt.legend(loc='upper left')
plt.show()


# In[27]:


enron_df_new['long_term_incentive'].argmax()


# In[28]:


enron_df_new.drop('MARTIN AMANDA K', axis = 0, inplace = True)


# In[29]:


# Scatterplot of restricted_stock vs 'restricted_stock_deferred' w.r.t POI

plt.scatter(enron_df_new['restricted_stock'][enron_df_new['poi'] == False],
            enron_df_new['restricted_stock_deferred'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(enron_df_new['restricted_stock'][enron_df_new['poi'] == True],
            enron_df_new['restricted_stock_deferred'][enron_df_new['poi'] == True],
            color = 'r', label = 'POI')

    
plt.xlabel('restricted_stock')
plt.ylabel('restricted_stock_deferred')
plt.title("Scatterplot of restricted_stock vs 'restricted_stock_deferred' w.r.t POI")
plt.legend(loc='upper right')
plt.show() 


# In[30]:


enron_df_new['restricted_stock_deferred'].argmax()


# In[31]:


enron_df_new.drop('BHATNAGAR SANJAY', axis = 0, inplace = True)


# In[32]:


plt.scatter(enron_df_new['from_poi_to_this_person'][enron_df_new['poi'] == False],
            enron_df_new['from_this_person_to_poi'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(enron_df_new['from_poi_to_this_person'][enron_df_new['poi'] == True],
            enron_df_new['from_this_person_to_poi'][enron_df_new['poi'] == True],
            color = 'r', label = 'POI')

    
plt.xlabel('from_poi_to_this_person')
plt.ylabel('from_this_person_to_poi')
plt.title("Scatterplot of count of from and to mails between poi and this_person w.r.t POI")
plt.legend(loc='upper right')
plt.show() 


# In[33]:


enron_df_new['fraction_mail_from_poi'] = enron_df_new['from_poi_to_this_person']/enron_df_new['from_messages'] 
enron_df_new['fraction_mail_to_poi'] = enron_df_new['from_this_person_to_poi']/enron_df_new['to_messages']

# Scatterplot of fraction of mails from and to between poi and this_person w.r.t POI
plt.scatter(enron_df_new['fraction_mail_from_poi'][enron_df_new['poi'] == False],
            enron_df_new['fraction_mail_to_poi'][enron_df_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(enron_df_new['fraction_mail_from_poi'][enron_df_new['poi'] == True],
            enron_df_new['fraction_mail_to_poi'][enron_df_new['poi'] == True],
            color = 'r', label = 'POI')

    
plt.xlabel('fraction_mail_from_poi')
plt.ylabel('fraction_mail_to_poi')
plt.title("Scatterplot of fraction of mails between poi and this_person w.r.t POI")
plt.legend(loc='upper right')
plt.show() 


# ## Preparing for Feature Processing

# In[34]:


# Clean all 'inf' values which we got if the person's from_messages = 0
enron_df_new = enron_df_new.replace('inf', 0)
enron_df_new = enron_df_new.fillna(0)

# Converting the above modified dataframe to a dictionary
enron_dict = enron_df_new.to_dict('index')
print "Features of modified data_dictionary:-"
print " Total number of datapoints: ",len(enron_dict)
print "Total number of features: ",len(enron_dict['METTS MARK'])


# In[45]:


# Store to my_dataset for easy export below.
my_dataset = enron_dict


# In[46]:


# Features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi" (target variable).

features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'bonus-to-salary_ratio', 'deferral_payments', 'expenses', 
                 'restricted_stock_deferred', 'restricted_stock', 'deferred_income','fraction_mail_from_poi', 'total_payments',
                 'other', 'fraction_mail_to_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 
                 'from_messages', 'shared_receipt_with_poi', 'loan_advances', 'director_fees', 'exercised_stock_options',
                'total_stock_value']


# In[47]:


# Extract features and labels from dataset for local testing
data = featureFormat(dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[48]:


# Split data into training and testing datasets
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, 
                                                              test_size=0.3, random_state=42)

# Stratified ShuffleSplit cross-validator
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state = 42)

# Importing modules for feature scaling and selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Defining features to be used via the pipeline
## 1. Feature scaling
scaler = MinMaxScaler()

## 2. Feature Selection
skb = SelectKBest(f_classif)


# In[49]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# In[50]:


# Classifier 1: Logistic Regression
lr_clf = LogisticRegression()

pipeline = Pipeline(steps=[("SKB", skb), ("LogisticRegression", lr_clf)])

param_grid = {"SKB__k": range(9, 10),
              'LogisticRegression__tol': [1e-2, 1e-3, 1e-4],
              'LogisticRegression__penalty': ['l1', 'l2']
             }

grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'f1')

t0 = time()
#clf = clf.fit(features_train, labels_train)
grid.fit(features, labels)
print "Training Time: ", round(time()-t0, 3), "s"

# Best algorithm
clf = grid.best_estimator_

t0 = time()
# Refit the best algorithm:
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "Testing time: ", round(time()-t0, 3), "s"

# Evaluation Measures
print "Accuracy of DT classifer is  : ", accuracy_score(labels_test, prediction)
print "Precision of DT classifer is : ", precision_score(prediction, labels_test)
print "Recall of DT classifer is    : ", recall_score(prediction, labels_test)
print "f1-score of DT classifer is  : ", f1_score(prediction, labels_test)


# In[51]:


# Classifier 2: KNN Classifier

clf_knn = KNeighborsClassifier()

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state = 42)
pipeline = Pipeline(steps = [("scaling", scaler), ("SKB", skb),  ("knn",clf_knn)])
param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18], 
              "knn__n_neighbors": [3,4,5,6,7,8,9,11,12,13,15],
              }

grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'f1')

t0 = time()
# clf = clf.fit(features_train, labels_train)
grid.fit(features, labels)
print "Training time: ", round(time()-t0, 3), "s"

# Best Algorithm
clf = grid.best_estimator_

t0 = time()
# Refit the best algorithm:
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "Testing time: ", round(time()-t0, 3), "s"

# Evaluation measures
print "Accuracy of DT classifer is  : ", accuracy_score(labels_test, prediction)
print "Precision of DT classifer is : ", precision_score(prediction, labels_test)
print "Recall of DT classifer is    : ", recall_score(prediction, labels_test)
print "f1-score of DT classifer is  : ", f1_score(prediction, labels_test)


# In[52]:


## Classifier 3: Gaussian Naive Bayes (GaussianNB) classifier

clf_gnb = GaussianNB()

pipeline = Pipeline(steps = [("SKB", skb), ("NaiveBayes", clf_gnb)])
param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}

grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'f1')

t0 = time()
grid.fit(features, labels)
print "Training time: ", round(time()-t0, 3), "s"

# Best Algorithm
clf = grid.best_estimator_

t0 = time()
# Refit the best algorithm:
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print"Testing time: ", round(time()-t0, 3), "s"

print "Accuracy of GaussianNB classifer is  : ", accuracy_score(labels_test, prediction)
print "Precision of GaussianNB classifer is : ", precision_score(prediction, labels_test)
print "Recall of GaussianNB classifer is    : ", recall_score(prediction, labels_test)
print "f1-score of GaussianNB classifer is  : ", f1_score(prediction, labels_test)


# In[53]:


# Obtaining the boolean list showing selected features
features_selected_bool = grid.best_estimator_.named_steps['SKB'].get_support()

# Finding the features selected by SelectKBest
features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
print "Total number of features selected by SelectKBest algorithm: ", len(features_selected_list)

# Finding the score of features 
feature_scores =  grid.best_estimator_.named_steps['SKB'].scores_

# Finding the score of features selected by selectKBest
feature_selected_scores = feature_scores[features_selected_bool]

# Creating a pandas dataframe and arranging the features based on their scores and ranking them 
imp_features_df = pd.DataFrame({'Features_Selected':features_selected_list, 'Features_score':feature_selected_scores})
imp_features_df.sort_values('Features_score', ascending = False, inplace = True)
Rank = pd.Series(list(range(1, len(features_selected_list)+1)))
imp_features_df.set_index(Rank, inplace = True)

print "The following table shows the feature selected along with its corresponding scores."
imp_features_df


# In[54]:


dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




