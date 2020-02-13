

SemEval2018 Task 2 Multilingual Emoji Prediction
=====

# **TASK**
* Subtask 1: Emoji Prediction in English
* Subtask 2: Emoji Prediction in Spanish

Official description:
https://competitions.codalab.org/competitions/17344#learn_the_details-overview

Our system description paper is here:
https://arxiv.org/abs/1805.10267

```
@inproceedings{jin-pedersen-2018-duluth,
    title = "{D}uluth {UROP} at {S}em{E}val-2018 Task 2: Multilingual Emoji Prediction with Ensemble Learning and Oversampling",
    author = "Jin, Shuning and Pedersen, Ted",
    booktitle = "Proceedings of The 12th International Workshop on Semantic Evaluation",
    year = "2018",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S18-1077",
}
```

# **AUTHORS**
Team: Duluth UROP :yum:
- Shuning Jin, University of Minnesota Duluth, jinxx596 AT d.umn.edu
- Ted Pedersen, University of Minnesota Duluth, tpederse AT d.umn.edu

# **Data**
#### test data
This is the official test data: `data/test`


#### training data

Due to Twitter's privacy policy, I cannot upload the training data. Please follow the official instruction to crawl the full training data from web: https://competitions.codalab.org/competitions/17344#learn_the_details-data


#### toy training data
For trial purpose,  this is a very small toy training data with 200 examples: `data/train_toy`



# **Script**
## 0. Start

Configuration: need python 3 environment, download required packages

```bash
pip install -r requirements.txt
```

Try demo runs: run the pipeline commands at once

```bash
bash script/es_demo1.sh
bash script/es_demo2.sh
bash script/us_demo1.sh
bash script/us_demo2.sh
```

## 1. Preprocessing

```bash
python preprocess.py \
--train_text [train_text] \
--train_label [train_key] \
--test_text [test_text] \
--run_dir [run_dir]
```

Example of usage:
```bash
python preprocess.py \
--train_text data/train_toy/es_train.text \
--train_label data/train_toy/es_train.labels \
--test_text data/test/es_test.text \
--run_dir demo
```

The script generates 3 files:
* experiment/[run_dir]/preprocess
  * test_x_dtm.npz
  * train_x_dtm.npz
  * train_y

## 2. Resampling

This step is optional, depending on which model to use next.

```bash
python sampling.py \
--run_dir [run_dir] \
--resample [resample_choice] \
--knn [knn, optional]
```

resample_choice:
- smote: for oversampling
- enn: for undersampling

knn:
- integer, 5: default, 1: for small examples
- optional, only used for smote



Example of usage:
```bash
python sampling.py \
--run_dir demo \
--resample smote \
--knn 1
```

The script generates 2 files:
* experiment/[run_dir]/preprocess
  * test_x_dtm\_[resample_choice].npz
  * train_y\_[resample_choice]



## 3. Classification
```bash
python model.py \
--run_dir [run_dir] \
--output [output_path] \
--model [model] \
--resample [resample_choice, optional] \
--weight_strategy [language, optional]
```

- model:
  - naive_bayes
  - logitstic_regression
  - random_forest
  - ensemble1
  - ensemble2
  - meta_ensemble

- language: es - Spanish, us - English

- resample
  - optional: only if resampled data is to be used (e.g. ensemble2, meta_ensemble)
  - smote, enn

Example of usage:
```bash
python model.py \
--run_dir demo \
--output es_output_meta \
--model meta_ensemble \
--resample smote \
--weight_strategy es
```

The script generates 1 file:
* experiment/[run_dir]/[output_path]

## 4. Evaluation

```bash
python scorer.py [gold_path] [output_path] [language(es/us)]
```
Language: es - Spanish, us - English

Example of usage:
```bash
python scorer.py \
data/test/es_test.labels \
experiment/es_output_meta \
es
```
