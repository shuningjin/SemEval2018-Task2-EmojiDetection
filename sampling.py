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
File: sampling.py
Resampling data

oversampling with SMOTE
store x in new sparse matrix
'''

import sys
from collections import Counter
import time
import os
import argparse

from scipy.sparse import load_npz, save_npz
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

from preprocess import save_label, save_sparse_matrix, load_label


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--run_dir", type=str, required=True, default=None, help="",)
    parser.add_argument("--resample", type=str, required=True,
                        default='none', choices=["smote", "enn", "none"], help="",)
    parser.add_argument("--knn", type=int, required=False, default=5, help="",)
    return parser.parse_args(cl_arguments)


def compare_frequency(train_y, y_resampled):
    """
    print label frequency before and after resampling
    """
    before = Counter(train_y)
    after = Counter(y_resampled)
    diff = set(before.keys()).difference(set(after.keys()))
    [after.update({i: 0}) for i in diff]

    classes, before = list(zip(*sorted(before.items())))
    classes, after = list(zip(*sorted(after.items())))

    layout = '|'.join(['{:<10s}'] + ['{:<7d}'] * len(classes))
    result = '\n'.join([layout.format("class", *classes),
                        layout.format("before", *before),
                        layout.format("after", *after)])
    print(result)


def sampling(type, train_x_dtm, train_y, k=5):
    if type == 'smote':
        print('Oversampling: {}'.format(type))
        model = SMOTE(random_state=0, k_neighbors=k)
    elif type == 'enn':
        print('Undersampling: {}'.format(type))
        model = EditedNearestNeighbours()  # random_state=0

    x_resampled, y_resampled = model.fit_sample(train_x_dtm, train_y)
    compare_frequency(train_y, y_resampled)

    return x_resampled, y_resampled


def main(runname, type, k=5):
    if type == 'none':
        return
    print('\n--- PHASE: RESAMPLING ---')
    preprocess_dir = os.path.join('experiment', runname, 'preprocess')
    files = os.listdir(preprocess_dir)
    target_x = 'train_x_dtm_{:s}.npz'.format(type)
    target_y = 'train_y_{:s}'.format(type)
    if target_x in files and target_y in files:
        print('Resampled files already exists. Pass this step.')
        return

    # read preprocessed files
    train_y = load_label(os.path.join(preprocess_dir, 'train_y'))
    train_x_dtm = load_npz(os.path.join(preprocess_dir, 'train_x_dtm.npz'))

    # resampling
    x_resampled, y_resampled = sampling(type, train_x_dtm, train_y, k)

    # save resampled files
    save_sparse_matrix(os.path.join(preprocess_dir, target_x), x_resampled)
    save_label(os.path.join(preprocess_dir, target_y), y_resampled)


if __name__ == "__main__":

    start_time = time.time()

    args = handle_arguments(sys.argv[1:])
    runname = args.run_dir  # train text, train label, test text
    resample_choice = args.resample       # 'smote', 'enn', 'none'

    main(runname, resample_choice, k=args.knn)

    seconds = time.time() - start_time
    minutes = seconds / 60
    print("Sampling time: {:.2f} seconds, {:.2f} minutes".format(seconds, minutes))
