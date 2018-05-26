Official evaluation script
=====

The official evaluation script for this task can be run from the terminal as follows:

```bash
python scorer-semeval2018-task2.py [gold_key] [output_key]
```

Example of usage:

```bash
python scorer-semeval2018-task2.py trial/english.trial.gold.txt trial/english.output.txt
```

*Note: Gold and output files should have the same number of lines. If for some occurrence the system is not confident enough to output any answer, just leave that line empty.*
_______________________________________________
SemEval2018 Task 2 Multilingual Emoji Prediction
=====

# **TASK**
* Subtask 1: Emoji Prediction in English
* Subtask 2: Emoji Prediction in Spanish
https://competitions.codalab.org/competitions/17344#learn_the_details-overview

# **AUTHORS**
Team: Duluth UROP
Shuning Jin, University of Minnesota Duluth, jinxx596@d.umn.edu
Ted Pedersen, University of Minnesota Duluth, tpederse@d.umn.edu

# **Script**
## 1. Preprocessing

```bash
python preprocess.py [train_text] [train_key] [test_text] [outname1]
```

Example of usage:
```bash
python preprocess.py es_train.text es_train.labels es_test.text es
```

The script generates 3 files:
* test_x_dtm_[outname1].npz
* train_x_dtm_[outname1].npz
* train_y_[outname1]

## 2. Resampling

```bash
python sampling.py [outname1] [choice]
```

Example of usage:
```bash
python sampling.py es 1
```

The script generates 2 files:
* test_x_dtm_[outname1]_[outname2].npz
* train_y_[outname1]_[outname2]

Choice: 1 Oversampling - SMOTE (Synthetic Minority Oversampling Technique), outname2 = smote
        2 Undersampling - ENN (Edited Nearest Neighbors), outnam2 = enn

## 3. Classification
```bash
python model.py [outname1] [output] [choice(1-6)] [language(es/en)] [outname2]
```
Choice: 1 - MNB, 2 - LR, 3 - RF, 4 - Ensemble1, 5 - Ensemble2, 6 - MetaEnsemble
Language: es - Spanish, en - English
(outname2 is optional, only if resampled data is to be used)

Example of usage:
```bash
python model.py es es_output6 6 es smote
```

The script generates 1 files:
* [output]

## 4. Evaluation

```bash
python scorer.py [gold_path] [output_path] [language(es/en)]
```
Language: es - Spanish, en - English

Example of usage:
```bash
python scorer.py es_test.labels es_output6 es
```
