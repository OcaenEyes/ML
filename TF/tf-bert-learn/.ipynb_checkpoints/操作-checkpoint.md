```shell
sudo python run_classifier.py \
--task_name=comment \                                                                         
--do_predict=true \
--data_dir=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/data \
--vocab_file=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2/vocab.txt \
--bert_config_file=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2/bert_config.json \
--init_checkpoint=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2/bert_model.ckpt \
--max_seq_length=128 \
--train_batch=32 \
--learning_rate=2e-5 \
--num_train_epochs=5.0 \
--output_dir=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/output/comment_1
```


```shell
sudo python run_classifier.py \
--task_name=comment \
--do_train=true \
--do_eval=true \
--data_dir=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/data \
--vocab_file=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2/vocab.txt \
--bert_config_file=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2/bert_config.json \
--init_checkpoint=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2/bert_model.ckpt \
--max_seq_length=128 \
--train_batch=32 \
--learning_rate=2e-5 \
--num_train_epochs=5.0 \
--output_dir=/Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/output/comment_0
```



```shell
sudo python freeze_graph.py \
-bert_model_dir /Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2 \
-model_dir /Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/output/comment_1 \
-max_seq_len 128 \
-num_labels 3
```