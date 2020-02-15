python preprocess.py \
--train_text data/all/es_train.text \
--train_label data/all/es_train.labels \
--test_text  data/test/es_test.text \
--run_dir es_run

python sampling.py \
--run_dir es_run \
--resample smote \
--knn 5

python model.py \
--run_dir es_run \
--output es_output_meta \
--model meta_ensemble \
--resample smote \
--weight_strategy es

python scorer.py \
data/test/es_test.labels \
experiment/es_run/es_output_meta \
es
