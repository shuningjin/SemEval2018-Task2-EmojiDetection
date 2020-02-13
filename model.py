#!/usr/bin/python
# -*- coding: UTF-8 -*-

###########################
# SemEval-2018 Task 2:
# Multilingual Emoji Detection
# Team: Duluth UROP
# Author: Shuning Jin
# Environment: Python 3.6
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
import time
import argparse
import os

from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from mlxtend.classifier import EnsembleVoteClassifier


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--preprocess_dir", type=str, required=True, default=None, help="",)
    parser.add_argument("--model", type=str, required=True, default=None, help="",
                        choices=['logistic_regression', 'naive_bayes', 'random_foreset',
                                 'ensemble1', 'ensemble2', 'meta_ensemble', ],)
    parser.add_argument("--output", type=str, required=True, default=None, help="",)
    parser.add_argument("--resample", type=str, required=False,
                        default="none", choices=["smote", "enn", "none"], help="",)
    parser.add_argument("--weight_strategy", type=str, required=False,
                        default="none", choices=["es", "en"], help="",)
    return parser.parse_args(cl_arguments)


# 1 multinomial Naive Bayes
model1 = MultinomialNB(alpha=0.5, fit_prior=True)
# 2 logistic regression
#   (n_jobs=-1, use all cores for multiprocessing)
model2 = LogisticRegression(random_state=0, n_jobs=-1, C=1, solver='lbfgs', penalty='l2')
# 3 random forest
#   (n_jobs=-1, use all cores for multiprocessing)
model3 = RandomForestClassifier(
    n_estimators=20, random_state=0, n_jobs=-1, criterion='gini')
# base ensemble (sklearn)
weight_base = [1, 1, 1]
weight_meta = [1, 1]  # initialize
voting = VotingClassifier(
    estimators=[('mnb', model1), ('logistic', model2), ('rf', model3)], voting='soft', weights=weight_base)


def apply_model(model, resample=0):
    if resample == 0:
        model.fit(train_x_dtm, train_y)
    elif resample == 1:
        model.fit(X_resampled, y_resampled)

    pred_y = model.predict(test_x_dtm)

    '''
    with open(outfile, 'w') as f:
        for p in pred_y:
            print(p, file=f)
    '''

    output = open(os.path.join('experiment', outfile), 'w')
    for p in pred_y:
        output.write(format("%d\n") % p)
    output.close()


# meta ensemble
def meta_ensemble():
    # ensemble learning (mlxtend)
    eclf1 = EnsembleVoteClassifier(clfs=[model1, model2, model3],
                                   weights=weight_base, voting='soft', refit=True)
    eclf2 = EnsembleVoteClassifier(clfs=[model1, model2, model3],
                                   weights=weight_base, voting='soft', refit=True)
    eclf1.fit(train_x_dtm, train_y)
    print('ensemble1 fitted.')
    eclf2.fit(X_resampled, y_resampled)
    print('ensemble2 fitted.')

    eclf3 = EnsembleVoteClassifier(
        clfs=[eclf1, eclf2], weights=weight_meta, voting='soft', refit=False)

    return eclf3


def set_weight(strategy):
    global weight_base, weight_meta
    if strategy == 'es':
        weight_base = [1.1, 1, 1]
        weight_meta = [3, 1]
    elif strategy == 'en':
        weight_base = [1.5, 6, 1]
        weight_meta = [4, 1]


def main(name, outname, choice, weight_strategy="none", name2="none"):
    global train_x_dtm, train_y, X_resampled, y_resampled, test_x_dtm, outfile
    outfile = outname

    # load preprocess
    with open(os.path.join('experiment', name, 'train_y'), 'r') as f:
        train_y = [int(str(line).replace('\n', '')) for line in f]
    test_x_dtm = load_npz(os.path.join('experiment', name, 'test_x_dtm.npz'))
    train_x_dtm = load_npz(os.path.join('experiment', name, 'train_x_dtm.npz'))

    # load preprocess + resample
    if name2 != 'none':
        X_resampled = load_npz(os.path.join('experiment', name, 'train_x_dtm_' + name2 + '.npz'))
        with open(os.path.join('experiment', name, 'train_y_' + name2), 'r') as f:
            y_resampled = [int(str(line).replace('\n', '')) for line in f]
        print(X_resampled.shape, len(y_resampled))

    if weight_strategy != 'none':
        # depend on language: es/en
        set_weight(weight_strategy)

    if choice == 'naive_bayes':
        apply_model(model1)  # multinomial naive bayes
    elif choice == 'logistic_regression':
        apply_model(model2)  # logistic regression
    elif choice == 'random_forest':
        apply_model(model3)  # random forest
    elif choice == 'ensemble1':
        apply_model(voting)  # ensemble1
    elif choice == 'ensemble2':
        apply_model(voting, resample=1)  # ensemble2 (resampling)
    elif choice == 'meta_ensemble':
        eclf3 = meta_ensemble()  # meta ensemble (ensemble1 + ensemble2)
        apply_model(eclf3)
    else:
        print('Error: illegal choice.')


if __name__ == "__main__":

    args = handle_arguments(sys.argv[1:])
    # predict label
    outfile = args.output
    # train text, train label, test text from preprocessing
    preprocess_dir = args.preprocess_dir
    resample = args.resample
    # classifier
    choice = args.model
    weight_strategy = args.weight_strategy

    start_time = time.time()

    main(preprocess_dir, outfile, choice, weight_strategy, resample)

    seconds = time.time() - start_time
    minutes = seconds / 60
    print("Modeling time: {:.2f} seconds, {:.2f} minutes".format(seconds, minutes))
