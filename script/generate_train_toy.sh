# extract first n lines of full training set as toy training set
num=$1
for i in 'es_train.labels' 'es_train.text' 'us_train.labels' 'us_train.text'
do head -n $num data/all/$i > data/train_toy/$i
done
