from predict_tool import remove_tool, preTest
from flask import Flask, render_template
from flask import request

app = Flask(__name__)


# 首页
@app.route('/')
def upload():
    return render_template('upload.html')


# 传图片
@app.route('/upload', methods=['GET', 'POST'])
def getUpload():
    # 获取图片文件，设置图片文件name=1
    fileType = request.form.get("type")
    img = request.form.get("str")
    print(fileType)
    # imagedata = base64.b64decode(img)  # 转码后得到的图片
    res = remove_tool([img], [fileType])
    return res


if __name__ == '__main__':
    preTest()  # 预加载模型
    app.run()  # 框架启动
