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
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import time
import argparse
import os


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_text", type=str, required=True, default=None, help="",)
    parser.add_argument("--train_label", type=str, required=True, default=None, help="",)
    parser.add_argument("--test_text", type=str, required=False, default=None, help="",)
    parser.add_argument("--preprocess_dir", type=str, required=True, default=None, help="",)
    return parser.parse_args(cl_arguments)


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
            line = line.strip('\n')
            line = line.strip(' ')
            text.append(line)
    return text


def main(infile1, infile2, infile3, outname):

    # vectorize data: bag of n-grams
    # feature: unigram + bigram, document frequency cutoff = 5
    tokenizer = nltk.word_tokenize
    vect = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenizer, min_df=5)

    train_x = read_text(infile1)
    vect.fit(train_x)
    train_x_dtm = vect.transform(train_x)

    test_x = read_text(infile3)
    test_x_dtm = vect.transform(test_x)

    # print len(vect.get_feature_names())
    # print train_x
    # print vect.get_feature_names()

    # save preprocessed files
    os.makedirs('experiment', exist_ok=True)
    preprocess_dir = outname
    os.makedirs(os.path.join('experiment', preprocess_dir), exist_ok=True)
    # save text x as sparse matrix
    save_npz(os.path.join('experiment', preprocess_dir, 'train_x_dtm'), train_x_dtm)
    save_npz(os.path.join('experiment', preprocess_dir, 'test_x_dtm'), test_x_dtm)
    # save label y
    train_y = [int(str(line).replace('\n', '')) for line in open(infile2, 'r')]
    y = open(os.path.join('experiment', preprocess_dir, 'train_y'), 'w')
    for line in train_y:
        print(line, file=y)


if __name__ == "__main__":

    args = handle_arguments(sys.argv[1:])
    infile1 = args.train_text
    infile2 = args.train_label
    infile3 = args.test_text
    outname = args.preprocess_dir

    start_time = time.time()

    main(infile1, infile2, infile3, outname)
    seconds = time.time() - start_time
    minutes = seconds / 60
    print("Preprocess time: {:.2f} seconds, {:.2f} minutes".format(seconds, minutes))
