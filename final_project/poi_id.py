#!/::usr/bin/python

from __future__ import division
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.decomposition import RandomizedPCA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first featuer must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
print data_dict
data_dict.pop( "TOTAL", 0 )

### Task 2: Remove outliers
from sklearn import preprocessing
sal = 0
Salary = []
count = []
count_num = 0
mode = 0
mil_gaya = []
j = 0
for i in data_dict.itervalues():
	if i['salary'] != 'NaN':
		Salary.append(i['salary'])
		sal = sal + i['salary']
		count_num = count_num + 1
		count.append(count_num)
		#count = count + 1
	if i['salary'] > 1000000 and i['salary'] != 'NaN':
		print "High Salary",i['salary']
		print "The Person is ::", i
		mil_gaya.append(j)
	j = j + 1
j = 0
for i in data_dict:
	
	if(j in mil_gaya):
		print "The Culprit",i
	j = j+1
	"""
plt.scatter(Salary, count)
plt.ylabel('Number')
plt.xlabel('Salary')
plt.show()
"""
"""mean =  sal / count
print "Mean =",mean

#to find mode
from collections import Counter
data = Counter(Salary)
mode = data.most_common()
print "Mode::", mode
"""
"""
===============================================================================
								NEW FEATURE LIST
===============================================================================
"""
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi','salary','total_payments','bonus'] 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
""""
==============================================================================
			USING PCA TO REDUCE THE DIMENSIONALITY OF THE FEATURES
==============================================================================
"""
"""
pca = RandomizedPCA(whiten=True).fit(features)
new_features = pca.transform(features)
===============================================================================
LASSO CLASSIFIER NOT WORKING AT ALL GIVING 100% RESULTS WHICH IS IMPOSSIBLE
==============================================================================
from sklearn import linear_model
clf = linear_model.Lasso(alpha = 1.0)


===============================================================================
                        CLASSIFIER 1:KNeighborsClassifier
===============================================================================
from sklearn.neighbors import	KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3,weights='distance',p=1)
--------------------------------------------------------------
ac = .82
preecision = .33
recall = .13
True postive = 261
--------------------------------------------------------------

--------------------------------------------------------------
===================================================================================
                             Classifier 2:GaussianNB
===================================================================================
"""
"""
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    
----------------------------------------------------------------------------------
Accuracy: 0.82669	
	Precision: 0.33550	
	Recall: 0.12900	
	True positives:  258
---------------------------------------------------------------------------------
=============================================================================
                        CLASSIFIER 3: kMEANS
=============================================================================

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
--------------------------------------------------------------
	Accuracy: 0.89754	
	Precision: 0.93739	
	Recall: 0.85933
	True positives: 5779

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=3)
---------------------------------------------------------
Accuracy: 0.81577
Precision: 0.61705
Recall: 0.47761
True positives: 1397
---------------------------------------------------------

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=7)
-------------------------------------------------------
Accuracy: 0.93838
Precision: 0.97485
Recall: 0.9302
True positives: 7945
-------------------------------------------------------

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=10)
-------------------------------------------------------------
Accuracy: 0.96069	
Precision: 0.98305
Recall: 0.96628
True positives: 9801	
------------------------------------------------------------


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=15)
-----------------------------------------------------------
****************************BEST RESULT*****************************
Accuracy: 0.97269
Precision: 0.98813
Recall: 0.98012
True positives: 10992
----------------------------------------------------------

  
==================================================================
							DECIOSN TREE
==================================================================

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 2)
------------------------------------------------------------------
Accuracy: 0.81946
Precision: 0.12527
Recall: 0.02900	
True positives:   58
------------------------------------------------------------------


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 2)
-----------------------------------------------------------------
Accuracy: 0.77046	
Precision: 0.26836
Recall: 0.28500
True positives:  570
----------------------------------------------------------------

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 5)
---------------------------------------------------------------
Accuracy: 0.78831
Precision: 0.29476
Recall: 0.27000
True positives:  540
--------------------------------------------------------------

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 10)
---------------------------------------------------------------
Accuracy: 0.79969	
Precision: 0.29372
Recall: 0.21500
True positives:  430
--------------------------------------------------------------

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 7)
--------------------------------------------------------------
Accuracy: 0.79938
Precision: 0.32589	
Recall: 0.28450
True positives:  569
---------------------------------------------------------------

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 6)
-------------------------------------------------------------
**************************BEST RESULT**********************
Accuracy: 0.79515
Precision: 0.31715	
Recall: 0.28750
True positives:  575	
----------------------------------------------------------------


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 8)
--------------------------------------------------------------
Accuracy: 0.80085	
Precision: 0.32248
Recall: 0.26750
True positives:  535
--------------------------------------------------------------

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 9)
--------------------------------------------------------------
	Accuracy: 0.80223	
	Precision: 0.31850	
	Recall: 0.25050	
	True positives:  501
---------------------------------------------------------------
===========================================================================
							k neighbour classifiers
=========================================================================

from sklearn.neighbors import	KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3,weights='uniform',p=1)
--------------------------------------------------------------------
Accuracy = 0.8301
Precision = 0.2808
Recall = 0.0660
True positives = 132
-----------------------------------------------------------------



from sklearn.neighbors import	KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3,weights='distance',p=1)
------------------------------------------------------------------
Accuracy: 0.82569
Precision: 0.33122
Recall: 0.13050
True positives:  261
-------------------------------------------------------------------

from sklearn.neighbors import	KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5,weights='distance',p=1)
---------------------------------------------------------------------
Accuracy: 0.83862	
Precision: 0.36158	
Recall: 0.06400
True positives:  128
--------------------------------------------------------------------


from sklearn.neighbors import	KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5,weights='distance',p=2)
---------------------------------------------------------------------
Accuracy: 0.83469	
Precision: 0.37221
Recall: 0.10850
True positives:  217	
---------------------------------------------------------------------

from sklearn.neighbors import	KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 6,weights='distance',p=2)
	Accuracy: 0.81615	
	Precision: 0.01970
	Recall: 0.00400
	True positives:    8	
----------------------------------------------------------------------
===============================================================================================================
								USING PCA AND OTHER CLASSIFIER THROUGH PIPELINE
===============================================================================================================

-----------------------------------------------------------------------------------------------------------------
                                            KNEIGHBOUR CLASSIFIER
 ----------------------------------------------------------------------------------------------------------------

from sklearn.neighbors import	KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors = 3,weights='distance',p=1)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()),('kNeigh', KNeighborsClassifier(n_neighbors = 3,weights='distance',p=1))]
clf = Pipeline(estimators)
----------------------------------------------------------------------------------------------------------------
	Accuracy: 0.82646	
	Precision: 0.33632
	Recall: 0.13150	
	rue positives:  263	
-----------------------------------------------------------------------------------------------------------------
												DECISION TREE
-----------------------------------------------------------------------------------------------------------------


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 6)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()),('Dtree', tree.DecisionTreeClassifier(min_samples_split = 6))]
clf = Pipeline(estimators)
Accuracy: 0.79315
Precision: 0.30635
Recall: 0.27250
True positives:  545

----------------------------------------------------------------------------------------------------------------------
													kMEANS
-------------------------------------------------------------------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()),('kMEANS',  KMeans(n_clusters=15))]
clf = Pipeline(estimators)
Accuracy: 0.97500	
Precision: 0.99069
Recall: 0.98014	
True positives: 10955

---------------------------------------------------------------------------------------------------------------------
														svm
---------------------------------------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()),('svm',SVC())]
clf = Pipeline(estimators)

--------------------------------------------------------------------------------------------------------------
													 GaussianNB
-----------------------------------------------------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()),('NB',  GaussianNB())]
clf = Pipeline(estimators)
Accuracy: 0.82800	
Precision: 0.32544
Recall: 0.11000
rue positives:  220


-----------------------------------------------------------------------------------------------------------------------

# Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.

cluster import KMeans
clf = KMeans(n_clusters=20)

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()),('kMEANS',  KMeans(n_clusters=15))]
clf = Pipeline(estimators)
"""

		
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    
	


test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
