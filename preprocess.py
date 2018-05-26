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
File: preprocess.py
Preprocessing text

normalize to lowercase
deal with punctuation, non-ASCII (remove or replace)
tokenize and vectorize
generate spase matrices
'''

import sys
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import time; start_time = time.time()

#---------------------------------------
def preprocess_x(infile):
    raw = ''
    input = open(infile,'r')
    for line in input:
        raw +=line
    # lowercase
    raw = raw.lower()
    # punctuation
    raw = raw.replace(r',',' ') #remove comma
    # non-ASCII characters
    # remove
    raw = raw.replace(r'…',' ') # '\xe2\x80\xa6', U+2026, horizontal ellipsis  # clean text
    raw = raw.replace(r'•',' ') # '\xe2\x80\xa2', U+2022, bullet
    raw = raw.replace(r'·',' ') # '\xc2\xb7', U+00B7, middle dot
    raw = raw.replace(r'・',' ') # '\xe3\x83\xbb', U+30FB, Katakana Middle Dot
    raw = raw.replace(r'，',' ') # '\xef\xbc\x8c'
    raw = raw.replace(r'—',' ') # '\xe2\x80\x94'
    raw = raw.replace(r'–',' ') # '\xe2\x80\x93'
    # replace with standard ASCII
    raw = raw.replace(r'’',"'") # '\xe2\x80\x99', U+2019, RIGHT SINGLE QUOTATION MARK
    raw = raw.replace(r'‘',"'") # '\xe2\x80\x98', U+2018, LEFT SINGLE QUOTATION MARK
    raw = raw.replace(r'“',r'"') # '\xe2\x80\x9c'
    raw = raw.replace(r'”',r'"') # '\xe2\x80\x9d'
    raw = raw.replace(r'！',r'!') # '\xef\xbc\x81'
    return raw

#---------------------------------------
def main(infile1,infile2,infile3,outname):
    train_y = [int(str(line).replace('\n','')) for line in open(infile2,'r')]
    raw = preprocess_x(infile1)
    raw2 = preprocess_x(infile3)
    train_x = re.findall(r'(.*)\n',raw)
    test_x = re.findall(r'(.*)\n',raw2)

    # vectorize data: bag of n-grams
    # feature: unigram + bigram, document frequency cutoff = 5
    tokenizer = nltk.word_tokenize
    vect = CountVectorizer(ngram_range=(1,2),tokenizer=tokenizer,min_df=5)
    vect.fit(train_x)
    train_x_dtm = vect.transform(train_x)
    test_x_dtm = vect.transform(test_x)
    #print len(vect.get_feature_names())
    #print train_x
    #print vect.get_feature_names()

    scipy.sparse.save_npz('train_x_dtm_'+outname, train_x_dtm)
    scipy.sparse.save_npz('test_x_dtm_'+outname, test_x_dtm)
    y = open('train_y_'+outname,'w')
    for line in train_y:
        print >> y, line
#end main
#---------------------------------------
if __name__ == "__main__":

    infile1= sys.argv[1] #train text
    infile2= sys.argv[2] #train label
    infile3 = sys.argv[3] #test text
    outname = sys.argv[4] #predict label
    main(infile1,infile2,infile3,outname)

    seconds = time.time() - start_time
    minutes = seconds/60
    print("--- %s seconds ---" %seconds)
    print("--- %s minutes ---" %minutes)
    sys.exit(1)
