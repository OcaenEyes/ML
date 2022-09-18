FROM python:3.6.5
MAINTAINER OCEANZY
RUN rm -rf /home/pythonapps/nlp/chatboot_web
RUN mkdir -p /home/pythonapps/nlp/chatboot_web
COPY requirements.txt /home/pythonapps/nlp/chatboot_web/
WORKDIR /home/pythonapps/nlp/chatboot_web
RUN pip install --upgrade pip -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
COPY . /home/pythonapps/nlp/chatboot_web
EXPOSE 8808
CMD ["python","server.py"]