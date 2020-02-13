#!/usr/bin/python
# -*- coding: UTF-8 -*-

###########################
# SemEval-2018 Task 2:
# Multilingual Emoji Detection
# Team: Duluth UROP
# Author: Shuning Jin
# Environment: Python 3.6
# Date: 2018-05-20
# Update: 2020-02-12
###########################

''' Description
File: preprocess.py
Preprocessing text

normalize to lowercase
deal with punctuation, non-ASCII (remove or replace)
tokenize and vectorize
generate scipy spase matrices
'''

import sys
import re
import time
import os
import argparse
from shutil import copyfile, copy
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_text", type=str, required=True, default=None, help="",)
    parser.add_argument("--train_label", type=str, required=True, default=None, help="",)
    parser.add_argument("--test_text", type=str, required=False, default=None, help="",)
    parser.add_argument("--run_dir", type=str, required=True, default=None, help="",)
    return parser.parse_args(cl_arguments)


def save_sparse_matrix(path, matrix):
    from scipy.sparse import save_npz
    save_npz(path, matrix)
    print('Save text as sparse matrix to: {:s}'.format(path))


def save_label(path, label_list):
    with open(path, 'w') as f:
        for line in label_list:
            print(line, file=f)
    print('Save labels to: {:s}'.format(path))


def load_label(path):
    label_list = [int(str(line).replace('\n', '')) for line in open(path, 'r')]
    print('Load labels from: {:s}'.format(path))
    return label_list


def read_text(path):
    """
    read raw text from path, preprocess, return a list of sentences
    """

    def clean(raw):
        # lowercase
        raw = raw.lower()
        # punctuation
        raw = raw.replace(r',', ' ')  # remove comma
        # non-ASCII characters
        # description: UTF-8 literal, unicode code point, name
        # remove
        raw = raw.replace(r'…', ' ')  # '\xe2\x80\xa6', U+2026, horizontal ellipsis  # clean text
        raw = raw.replace(r'•', ' ')  # '\xe2\x80\xa2', U+2022, bullet
        raw = raw.replace(r'·', ' ')  # '\xc2\xb7', U+00B7, middle dot
        raw = raw.replace(r'・', ' ')  # '\xe3\x83\xbb', U+30FB, Katakana middle dot
        raw = raw.replace(r'，', ' ')  # '\xef\xbc\x8c', U+FF0C, fullwidth comma
        raw = raw.replace(r'—', ' ')  # '\xe2\x80\x94', U+2014, EM dash
        raw = raw.replace(r'–', ' ')  # '\xe2\x80\x93', U+2013, EN dash
        # replace with standard ASCII
        raw = raw.replace(r'’', "'")  # '\xe2\x80\x99', U+2019, right single quotation mark
        raw = raw.replace(r'‘', "'")  # '\xe2\x80\x98', U+2018, left single quotation mark
        raw = raw.replace(r'“', r'"')  # '\xe2\x80\x9c', U+201C, left double quotation mark
        raw = raw.replace(r'”', r'"')  # '\xe2\x80\x9d', U+201D, right double quotation mark
        raw = raw.replace(r'！', r'!')  # '\xef\xbc\x81', U+FF01, fullwidth exclamation mark
        return raw

    text = []
    with open(path, 'r') as f:
        for line in f:
            line = clean(line)
            line = line.strip('\n').strip(' ')
            text.append(line)
    print('Read text: {:s}. Total example: {:d}'
          .format(path, len(text)))
    return text


def main(train_text_path, train_label_path, test_text_path, runname):
    print('\n--- PHASE: PREPROCESSING ---')
    # file logic
    run_dir = os.path.join('experiment', runname)
    preprocess_dir = os.path.join(run_dir, 'preprocess')
    os.makedirs('experiment', exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)
    files = os.listdir(preprocess_dir)
    if 'train_x_dtm.npz' in files and 'test_x_dtm.npz' in files and 'train_y' in files:
        print('Preprocessed files already exists. Pass this step.')
        return

    # vectorize data: bag of n-grams
    # feature: unigram + bigram, document frequency cutoff = 5
    tokenizer = nltk.word_tokenize
    vect = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenizer, min_df=5)

    train_x = read_text(train_text_path)
    vect.fit(train_x)
    train_x_dtm = vect.transform(train_x)

    test_x = read_text(test_text_path)
    test_x_dtm = vect.transform(test_x)

    # print len(vect.get_feature_names())
    # print train_x
    # print vect.get_feature_names()

    # save text x as sparse matrix
    save_sparse_matrix(os.path.join(preprocess_dir, 'train_x_dtm.npz'), train_x_dtm)
    save_sparse_matrix(os.path.join(preprocess_dir, 'test_x_dtm.npz'), test_x_dtm)
    # save label y
    copy(src=train_label_path, dst=os.path.join(preprocess_dir, 'train_y'))
    #train_y = load_label(train_label_path)
    #save_label(os.path.join(preprocess_dir, 'train_y'), train_y)


if __name__ == "__main__":

    args = handle_arguments(sys.argv[1:])
    train_text_path = args.train_text
    train_label_path = args.train_label
    test_text_path = args.test_text
    runname = args.run_dir

    start_time = time.time()

    main(train_text_path, train_label_path, test_text_path, runname)
    seconds = time.time() - start_time
    minutes = seconds / 60
    print("Preprocess time: {:.2f} seconds, {:.2f} minutes".format(seconds, minutes))
