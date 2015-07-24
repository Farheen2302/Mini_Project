#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( "TOTAL", 0 )
count = 0
c = 0
for i in data_dict.itervalues():
	count = count + 1
	if i['salary'] > 1000000.0 and i['salary'] != 'NaN' and  i['bonus'] == 7000000:
		print i
		c = count
count = 0
for i in data_dict:
	count = count + 1
	if count == c:
		print "name",i

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below



for point in data:
    salary = point[0]
    bonus = point[1]
    if salary > 25000000:
   		 print salary
   		 print bonus
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()