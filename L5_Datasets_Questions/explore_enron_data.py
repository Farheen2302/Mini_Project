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

count = 0
for i in enron_data.itervalues():
	if i["poi"] == True:
		count = count + 1
print count
"""


#print enron_data["PRENTICE JAMES"]["total_stock_value"]

#print enron_data["SKILLING JEFFREY K"]["poi"]
#print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

#print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
#print enron_data["SKILLING JEFFREY K"]["total_payments"]

#print enron_data["LAY KENNETH L"]["total_payments"]
#print enron_data["FASTOW ANDREW S"]["total_payments"]
#print enron_data.keys()
#print enron_data["FASTOW ANDREW S"]

"""count = 0
em = 0
for i in enron_data.itervalues():
	if i["salary"] != 'NaN' :
		count = count + 1
	if i["email_address"] != 'NaN' :
		em = em + 1
print "salary", count
print "mail" ,em


count = 0
for i in enron_data.itervalues():
	if i["total_payments"] == 'NaN' and i:
		count = count + 1
	
print "numof NaN", count
totat = (len(enron_data))
perc = (count*100) /totat
print "Percentage" , perc

"""


"""print enron_data["FASTOW ANDREW S"]
count = 0
em = 0
for i in enron_data.itervalues():
	if i["salary"] != 'NaN' :
		count = count + 1
	if i["email_address"] != 'NaN' :
		em = em + 1
print "salary", count
print "mail" ,em
"""


"""count = 0
p = 0
for i in enron_data.itervalues():
	if i["total_payments"] == 'NaN' and i["poi"] == True:
		count = count + 1
	if i["poi"] == True:
		p = p + 1
	
print "total", count,"POI+true",p
totat = (len(enron_data))
perc = (count*100) /p
print "Percentage" , perc
"""

print len(enron_data)

