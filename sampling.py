#!/usr/bin/python
# -*- coding: UTF-8 -*-

###########################
# SemEval-2018 Task 2:
# Multilingual Emoji Detection
# Team: Duluth UROP
# Author: Shuning Jin
# Environment: Python 2.7
# Date: 2018-05-20
###########################

''' Description
File: sampling.py
Resampling data

oversampling with SMOTE
store x in new sparse matrix
'''

import sys
import scipy.sparse
from collections import Counter
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
import time; start_time = time.time()

def Sampling(model):
    X_resampled, y_resampled = model.fit_sample(train_x_dtm, train_y)
    print(sorted(Counter(y_resampled).items()))

    scipy.sparse.save_npz('train_x_dtm_'+name+'_'+type, X_resampled)
    y_array = open('train_y_'+name+'_'+type,'w')
    for line in y_resampled:
        print >> y_array, line

smote = SMOTE(random_state=0)
enn = EditedNearestNeighbours(random_state=0)

#---------------------------------------
def main(name1,choice):
    global train_y, train_x_dtm, type, name
    name = name1

    train_y = [int(str(line).replace('\n','')) for line in open('train_y_'+name,'r')]
    train_x_dtm = scipy.sparse.load_npz('train_x_dtm_'+name+'.npz')

    if choice =='1':
        type = 'smote'
        Sampling(smote)
    elif choice == '2':
        type = 'enn'
        Sampling(enn)
#end main
#---------------------------------------
if __name__ == "__main__":

    name1= sys.argv[1] #train text, train label, test text
    choice = sys.argv[2]
    main(name1,choice)

    seconds = time.time() - start_time
    minutes = seconds/60
    print("--- %s seconds ---" %seconds)
    print("--- %s minutes ---" %minutes)
    sys.exit(1)
