# logistic_regression

python preprocess.py \
--train_text data/train_toy/es_train.text \
--train_label data/train_toy/es_train.labels \
--test_text  data/test/es_test.text \
--run_dir es_demo

python model.py \
--run_dir es_demo \
--output es_output_lr \
--model logistic_regression

python scorer.py \
data/test/es_test.labels \
experiment/es_demo/es_output_lr \
es
