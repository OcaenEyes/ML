#sudo rm -rf /Users/gaozhiyong/Documents/GitHub/ML/AI训练赛/情绪分析/output/*
export BERT_BASE_DIR=/Users/gaozhiyong/Documents/GitHub/ML/AI训练赛/情绪分析/uncased_L-2_H-128_A-2
export DATA_DIR=/Users/gaozhiyong/Documents/GitHub/ML/AI训练赛/情绪分析/data
export OUTPUT_DIR=/Users/gaozhiyong/Documents/GitHub/ML/AI训练赛/情绪分析/output
sudo python run_ugcclassifier.py \
--task_name=ugc \
--do_train=true \
--do_predict=true \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--data_dir=$DATA_DIR \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=5.0 \
--output_dir=$OUTPUT_DIR

# win 命令
python run_ugcclassifier.py ^
--task_name=ugc ^
--do_train=true ^
--do_predict=true ^
--vocab_file=D:\GitHub\ML\AI训练赛\情绪分析\model\chinese_L-12_H-768_A-12\vocab.txt ^
--bert_config_file=D:\GitHub\ML\AI训练赛\情绪分析\model\chinese_L-12_H-768_A-12\bert_config.json ^
--data_dir=D:\GitHub\ML\AI训练赛\情绪分析\data ^
--init_checkpoint=D:\GitHub\ML\AI训练赛\情绪分析\model\chinese_L-12_H-768_A-12\bert_model.ckpt ^
--max_seq_length=128 ^
--train_batch_size=10 ^
--learning_rate=2e-5 ^
--num_train_epochs=5.0 ^
--output_dir=D:\GitHub\ML\AI训练赛\情绪分析\output

#D:\\GitHub\\ML\\AI训练赛\\情绪分析\\output\\ugc\\versions