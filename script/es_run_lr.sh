python preprocess.py \
--train_text data/all/es_train.text \
--train_label data/all/es_train.labels \
--test_text  data/test/es_test.text \
--run_dir es_run

python model.py \
--run_dir es_run \
--output es_output_lr \
--model logistic_regression \

python scorer.py \
data/test/es_test.labels \
experiment/es_run/es_output_lr \
es
