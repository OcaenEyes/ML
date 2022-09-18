from flask import Flask, request, jsonify, render_template
import jieba
import time
import threading
import execute

"""
定义心跳检测函数
"""


def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    _timer = threading.Timer(60, heartbeat)
    _timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()

app = Flask(__name__, static_url_path="/static", template_folder="web/templates", static_folder="web/static")


@app.route("/message", methods=["post"])
def reply():
    # req_msg = request.json["send"]
    req_msg = request.form["send"]
    req_msg = " ".join(jieba.cut(req_msg))

    res_msg = execute.predict(req_msg)
    # 将unk值的词用微笑符号代替
    res_msg = res_msg.replace("_UNK", "^_^")
    res_msg = res_msg.replace(" ", "")
    res_msg = res_msg.strip()

    if res_msg == "":
        res_msg = "来和我聊聊天吧"

    return jsonify({"reply": res_msg})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8808)
