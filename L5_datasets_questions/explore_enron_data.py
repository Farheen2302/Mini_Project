#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
"""print len(enron_data.keys())
k= max(enron_data, key=lambda k: len(enron_data[k]))
print len(enron_data[k])
print max(len(v) for v in enron_data.itervalues())
"""
count = 0
for i in enron_data.itervalues():
	if i["poi"] == True:
		count = count + 1
print count

#print enron_data["SKILLING JEFFREY K"]["poi"]


