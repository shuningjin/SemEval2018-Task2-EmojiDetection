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

from preprocess import save_label, load_label


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--run_dir", type=str, required=True, default=None, help="",)
    parser.add_argument("--model", type=str, required=True, default=None, help="",
                        choices=['logistic_regression', 'naive_bayes', 'random_foreset',
                                 'ensemble1', 'ensemble2', 'meta_ensemble', ],)
    parser.add_argument("--output", type=str, required=True, default=None, help="",)
    parser.add_argument("--resample", type=str, required=False,
                        default="none", choices=["smote", "enn", "none"], help="",)
    parser.add_argument("--weight_strategy", type=str, required=False,
                        default="none", choices=["es", "us"], help="",)
    return parser.parse_args(cl_arguments)


''' base model '''
# 1 multinomial Naive Bayes
mnb = MultinomialNB(alpha=0.5, fit_prior=True)
# 2 logistic regression
#   (n_jobs=-1, use all cores for multiprocessing)
lr = LogisticRegression(random_state=0, n_jobs=-1, C=1, solver='lbfgs', penalty='l2')
# 3 random forest
#   (n_jobs=-1, use all cores for multiprocessing)
rf = RandomForestClassifier(
    n_estimators=20, random_state=0, n_jobs=-1, criterion='gini')

''' ensemble model '''
# initialize weights
weight_base = [1, 1, 1]
weight_meta = [1, 1]


def set_weight(strategy):
    # depend on language: es/en
    global weight_base, weight_meta
    if strategy == 'es':
        weight_base = [1.1, 1, 1]
        weight_meta = [3, 1]
    elif strategy == 'us':
        weight_base = [1.5, 6, 1]
        weight_meta = [4, 1]


# base ensemble (sklearn)
voting = VotingClassifier(
    estimators=[('mnb', mnb), ('logistic', lr), ('rf', rf)], voting='soft', weights=weight_base)


# meta ensemble
def meta_ensemble_model():
    # ensemble learning (mlxtend)
    ensemble1 = EnsembleVoteClassifier(clfs=[mnb, lr, rf],
                                       weights=weight_base, voting='soft', refit=True)
    ensemble2 = EnsembleVoteClassifier(clfs=[mnb, lr, rf],
                                       weights=weight_base, voting='soft', refit=True)
    meta_ensemble = EnsembleVoteClassifier(
        clfs=[ensemble1, ensemble2], weights=weight_meta, voting='soft', refit=False)

    ensemble1.fit(train_x_dtm, train_y)
    print('ensemble1 fitted.')
    ensemble2.fit(x_resampled, y_resampled)
    print('ensemble2 fitted.')

    return meta_ensemble


def apply_model(model, resample=0):
    if resample == 0:
        model.fit(train_x_dtm, train_y)
    elif resample == 1:
        model.fit(x_resampled, y_resampled)

    pred_y = model.predict(test_x_dtm)
    save_label(os.path.join(outfile), pred_y)


def main(runname, outname, choice, weight_strategy="none", resample="none"):
    print('\n--- PHASE: MODELING ---')
    global train_x_dtm, train_y, x_resampled, y_resampled, test_x_dtm, outfile

    preprocess_dir = os.path.join('experiment', runname, 'preprocess')
    outfile = os.path.join('experiment', runname, outname)

    # load preprocess
    train_y = load_label(os.path.join(preprocess_dir, 'train_y'))
    test_x_dtm = load_npz(os.path.join(preprocess_dir, 'test_x_dtm.npz'))
    train_x_dtm = load_npz(os.path.join(preprocess_dir, 'train_x_dtm.npz'))

    # load preprocess + resample
    if resample != 'none':
        x_resampled = load_npz(os.path.join(preprocess_dir, 'train_x_dtm_' + resample + '.npz'))
        y_resampled = load_label(os.path.join(preprocess_dir, 'train_y_' + resample))
        # print(x_resampled.shape, len(y_resampled))

    if weight_strategy != 'none':
        set_weight(weight_strategy)

    # multinomial naive bayes
    if choice == 'naive_bayes':
        apply_model(mnb)
    # logistic regression
    elif choice == 'logistic_regression':
        apply_model(lr)
    # random forest
    elif choice == 'random_forest':
        apply_model(rf)
    # ensemble1
    elif choice == 'ensemble1':
        apply_model(voting)
    # ensemble2 (resampling)
    elif choice == 'ensemble2':
        apply_model(voting, resample=1)
    # meta ensemble (ensemble1 + ensemble2)
    elif choice == 'meta_ensemble':
        meta_ensemble = meta_ensemble_model()
        apply_model(meta_ensemble)
    else:
        print('Error: illegal choice.')


if __name__ == "__main__":

    args = handle_arguments(sys.argv[1:])
    # predict label
    outfile = args.output
    # train text, train label, test text from preprocessing
    runname = args.run_dir
    resample = args.resample
    # classifier
    choice = args.model
    weight_strategy = args.weight_strategy

    start_time = time.time()

    main(runname, outfile, choice, weight_strategy, resample)

    seconds = time.time() - start_time
    minutes = seconds / 60
    print("Modeling time: {:.2f} seconds, {:.2f} minutes".format(seconds, minutes))
