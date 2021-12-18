FROM python:3.6.5
MAINTAINER OCEANZY
RUN rm -rf /home/pythonapps/nlp/poetry_apiserver
RUN mkdir -p /home/pythonapps/nlp/poetry_apiserver
COPY requirements.txt /home/pythonapps/nlp/poetry_apiserver/
WORKDIR /home/pythonapps/nlp/poetry_apiserver
RUN pip install --upgrade pip -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torch==1.10.0 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . /home/pythonapps/nlp/poetry_apiserver
EXPOSE 5002
CMD ["python","poetry_server.py"]