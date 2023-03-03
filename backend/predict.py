import joblib
import pandas as pd
import json
from flask import Flask, jsonify,request, render_template
import os
from flask_cors import CORS
import requests

app = Flask(__name__,
            static_folder='./static',
            template_folder="./dist")
CORS(app)

# read flask config
with open('flask_config.json','r',encoding='utf8')as fp:
    opt = json.load(fp)
    print('Flask Config : ', opt)

model_path=os.path.join(opt['model_dir'],opt['model_name'])
name={0:"DILI",1:"AIH"}

# input_url="http://127.0.0.1:5000/predict?nid=hkasdjaskdn&ALT=1311&GLB=21&PLT=246&ANA=0&IgG=1151"
# TODO 模型需要換一下

@app.route('/predict')
def profile():
    nid=request.args.get('id')
    ALT=request.args.get('ALT')
    GLB=request.args.get('GLB')
    PLT=request.args.get('PLT')
    ANA=request.args.get('ANA')
    IgG=request.args.get('IgG')
    print("debug_____________",nid)
    
    joblib_model = joblib.load(model_path)
    d={"ALT":ALT,"PLT":PLT,"GLO":GLB,"IgG":IgG, "ANA":ANA}
    result=joblib_model.predict_proba(pd.DataFrame(d,index=[0]))
    temp=0 if result[0][0]>result[0][1] else 1
    
    json=jsonify(nid=nid, result=name.get(temp), prob=round(result[0][temp],4))
    print("传输成功")
    return json

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get('http://localhost:8080/{}'.format(path)).text
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


