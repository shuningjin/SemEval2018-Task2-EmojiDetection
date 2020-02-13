# logistic_regression

python preprocess.py \
--train_text data/train_toy/es_train.text \
--train_label data/train_toy/es_train.labels \
--test_text  data/test/es_test.text \
--run_dir demo

python model.py \
--run_dir demo \
--output es_output_lr \
--model logistic_regression

python scorer.py \
data/test/es_test.labels \
experiment/demo/es_output_lr \
es
