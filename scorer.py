#!/usr/bin/python
# -*- coding: UTF-8 -*-

# adpated from scorer_semeval18.py, detailed_results.py
# https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/tree/master/tools
# evaulate accuracy, macro F/P/R
# command line: python scorer.py [gold_path] [pred_path] [language(es/en)]
#   eg. python scorer.py es_test.labels es_output6 es
#   eg. python scorer.py us_test.labels us_output6 en

''' Sample output:
Macro F-Score (official): 20.211
-----
Macro Precision: 21.228
Macro Recall: 21.117
Accuracy: 36.746
'''

import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score


def main(gold_path, pred_path, language):

    G = open(gold_path, 'r').read().replace("\r", "").replace(" ", "").split("\n")
    P = open(pred_path, 'r').read().replace("\r", "").replace(" ", "").split("\n")

    if language == 'es':
        n_labels = 19
    elif language == 'en':
        n_labels = 20
    labels = [str(x) for x in range(n_labels)]

    acc = accuracy_score(G, P)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        G, P, labels=labels, average="macro")
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        G, P, labels=labels, average="micro")

    print("Macro F-Score (official): " + str(round(f1_macro*100, 3)))
    print("-----")
    print("Macro Precision: " + str(round(p_macro*100, 3)))
    print("Macro Recall: " + str(round(r_macro*100, 3)))
    print("Accuracy: " + str(round(acc*100, 3)))


if __name__ == "__main__":
    gold_path = sys.argv[1]
    pred_path = sys.argv[2]
    language = sys.argv[3]
    main(gold_path, pred_path, language)
