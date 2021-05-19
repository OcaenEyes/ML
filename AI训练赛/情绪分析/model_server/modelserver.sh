echo '正在启动bert model_server'
cd /Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/model_server
sudo rm -rf tmp*

bert-base-serving-start \
    -model_dir /Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/output/comment_0 \
    -bert_model_dir /Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/uncased_L-2_H-128_A-2 \
    -model_pb_dir /Users/gaozhiyong/Documents/GitHub/ML/TF/tf-bert-learn/output/comment_0 \
    -mode CLASS \
    -max_seq_len 128 \
    -http_port 8091 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1