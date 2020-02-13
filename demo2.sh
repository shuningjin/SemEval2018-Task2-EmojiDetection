# meta_ensemble

python preprocess.py \
--train_text data/train_toy/es_train.text \
--train_label data/train_toy/es_train.labels \
--test_text  data/test/es_test.text \
--run_dir demo

python sampling.py \
--run_dir demo \
--resample smote \
--knn 1

python model.py \
--run_dir demo \
--output es_output_meta \
--model meta_ensemble \
--resample smote \
--weight_strategy es

python scorer.py \
data/test/es_test.labels \
experiment/demo/es_output_meta \
es
