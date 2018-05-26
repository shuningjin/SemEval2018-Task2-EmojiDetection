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
File: model.py
Perform classification

classifiers:
- Base: MNB (Multinomial Naive Bayes), LR (Logistic Regression), RF (Random Forest)
- Ensemble: MNB + LR + RF
    - Ensemble1 -> ensemble for original data
    - Ensemble2 -> ensemble for resampled data
- Meta Ensemble : Ensemble1  + Ensemble2

'''

import sys
import scipy.sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import time; start_time = time.time()

def apply_model(model,resample= 0):
    if resample == 0:
        model.fit(train_x_dtm, train_y)
    elif resample ==1:
        model.fit(X_resampled, y_resampled)

    pred_y= model.predict(test_x_dtm)

    output = open(outfile,'w')
    for p in pred_y:
        output.write(format("%d\n")%p)
    output.close()

#---------------------------------------
# 1 multinomial Naive Bayes
model1 = MultinomialNB(alpha=0.5, fit_prior=True)
# 2 logistic regression
#   (n_jobs=-1, use all cores for multiprocessing)
model2 = LogisticRegression(random_state=0,n_jobs=-1, C=1,solver = 'lbfgs',penalty = 'l2')
# 3 random forest
#   (n_jobs=-1, use all cores for multiprocessing)
model3 = RandomForestClassifier (n_estimators=20, random_state=0, n_jobs=-1, criterion='gini')

# base ensemble (sklearn)
weight_base = [1,1,1]; weight_meta = [1,1] # initialize
voting = VotingClassifier(estimators=[('mNB', model1), ('logistic', model2),('rf', model3)], voting='soft', weights=weight_base)

# meta ensemble
def meta_ensemble():
    #ensemble learning (mlxtend)
    eclf1 = EnsembleVoteClassifier(clfs=[model1, model2, model3], weights=weight_base, voting = 'soft',refit=True)
    eclf1.fit(train_x_dtm, train_y)
    print 'ensemble1 fitted.'

    eclf2 = EnsembleVoteClassifier(clfs=[model1, model2, model3], weights=weight_base, voting = 'soft',refit=True)
    eclf2.fit(X_resampled, y_resampled)
    print 'ensemble2 fitted.'

    eclf3 = EnsembleVoteClassifier(clfs=[eclf1,eclf2], weights=weight_meta, voting = 'soft',refit=False)
    apply_model(eclf3)

def set_weight (strategy):
    global weight_base,weight_meta
    if strategy == 'es' :
        weight_base = [1.1,1,1]
        weight_meta = [3,1]
    elif strategy == 'en':
        weight_base = [1.5,6,1]
        weight_meta = [4,1]

#---------------------------------------
def main(name,outname,choice,*args):
    global train_x_dtm,train_y,X_resampled,y_resampled,test_x_dtm,outfile
    outfile = outname

    train_y = [int(str(line).replace('\n','')) for line in open('train_y_'+name,'r')]
    test_x_dtm = scipy.sparse.load_npz('test_x_dtm_'+name+'.npz')
    train_x_dtm = scipy.sparse.load_npz('train_x_dtm_'+name+'.npz')

    if len(args)>=1:
        weight_strategy = args[0]   # depend on language: es/en
        set_weight (weight_strategy)
        if len(args)>=2:
            name2 = args[1] # train text, train label from resampling: smote
            X_resampled = scipy.sparse.load_npz('train_x_dtm_'+name+'_'+name2+'.npz')
            y_resampled = [int(str(line).replace('\n','')) for line in open('train_y_'+name +'_'+name2,'r')]
            print X_resampled.shape, len(y_resampled)

    #----------------------
    if choice =='1':
        apply_model(model1) #multinomial naive bayes
    elif choice == '2':
        apply_model(model2) #logistic regression
    elif choice == '3':
        apply_model(model3) #random forest
    elif choice =='4':
        apply_model(voting) #ensemble1
    elif choice == '5':
        apply_model(voting,resample=1) #ensemble2 (resampling)
    elif choice == '6':
        meta_ensemble() #meta ensemble (ensemble1 + ensemble2)
    else:
        print 'Error: illegal choice.'
# end main
#---------------------------------------
if __name__ == "__main__":

    name= sys.argv[1] # train text, train label, test text from preprocessing
    outfile = sys.argv[2] # predict label
    choice = sys.argv[3]   # classifier
    args = sys.argv[4:]
    main(name,outfile,choice,args)

    seconds = time.time() - start_time
    minutes = seconds/60
    print("--- %s seconds ---" %seconds)
    print("--- %s minutes ---" %minutes)
    sys.exit(1)
