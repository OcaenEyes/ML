docker cp /Users/gaozhiyong/Desktop/models f0112a005a20:models ##拷贝model文件到容器

docker run -p 8501:8501 -v /Users/gaozhiyong/Desktop/models:/models -t tensorflow/serving:1.13.0 --model_config_file=/models/model.config  # 启动docker部署

docker run -p 8501:8501 --mount type=bind,source=/Users/gaozhiyong/Desktop/models,target=/models -t tensorflow/serving:1.13.0 --model_config_file=/models/model.config  # 启动docker部署