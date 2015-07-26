#!/usr/bin/python

import os
import pickle
import re
import sys
import numpy as np
from scipy import sparse
sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []
sw = ["sara", "shackleton", "chris", "germani"]
total = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
       # if total < 5:
        #total = total + 1
        path = os.path.join('..', path[:-1])

        email = open(path, "r")
        text = str(parseOutText(email))
        for rm in sw:
            if(rm in text):
               text = text.replace(rm,"")
        
        word_data.append(text)

        if name == "Sara":
            from_data.append(0)
        if name == "Chris":
            from_data.append(1)

        email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )



from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(stop_words="english")
vec_fit=vec.fit_transform(word_data)
print len(vec.get_feature_names())
vec_words = vec.get_feature_names()
print vec_words[34597]
### in Part 4, do TfIdf vectorization here


