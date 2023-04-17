FROM python:3.6.5
MAINTAINER oceangzy
RUN rm -rf /home/pythonapps/nlp/gushi_server
RUN mkdir -p /home/pythonapps/nlp/gushi_server
COPY requirements.txt /home/pythonapps/nlp/gushi_server/
WORKDIR /home/pythonapps/nlp/gushi_server
RUN pip install --upgrade pip -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install tensorflow-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . /home/pyCOPY
EXPOSE 5003
CMD ["python","poetry_api.py"]