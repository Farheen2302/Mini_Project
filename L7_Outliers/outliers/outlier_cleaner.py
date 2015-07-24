#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []
    clean_data = []

    ### your code goes here
    err = []
   
    n = len(predictions)
    for i in range(n):
        e = abs(predictions[i] - net_worths[i])
        err.append(e)
        i = i + 1
   # print e
   # print ages
   # print net_worths
    clean_data = zip(ages,net_worths,err)
    """ for i in range(1):
        print clean_data[i]
    """
    
    clean_data.sort(key=lambda tup: tup[2])
    i = 0
    for i in range(len(clean_data) * (90) / 100 ):
        w = clean_data[i]
        cleaned_data.append(w)


    
    return cleaned_data

