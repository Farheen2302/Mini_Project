"""Create and train a Naive Bayes classifier in naive_bayes/nb_author_id.py. Use it to make predictions for the test set. What is the accuracy?
"""


#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#test
def nb_text_test():
	
	
	clf = GaussianNB()
	t0 = time()
	clf.fit(features_train,labels_train)
	print "training time:", round(time()-t0, 3), "s"
	t0 = time()
	pred = clf.predict(features_test)
	print "prediction time:", round(time()-t0, 3), "s"
	accuracy = accuracy_score(pred,labels_test)
	return accuracy
#########################################################

sc = nb_text_test()
print sc
