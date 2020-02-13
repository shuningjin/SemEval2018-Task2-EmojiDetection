# logistic_regression

python preprocess.py \
--train_text data/train_toy/us_train.text \
--train_label data/train_toy/us_train.labels \
--test_text  data/test/us_test.text \
--run_dir us_demo

python model.py \
--run_dir us_demo \
--output us_output_lr \
--model logistic_regression

python scorer.py \
data/test/us_test.labels \
experiment/us_demo/us_output_lr \
us
