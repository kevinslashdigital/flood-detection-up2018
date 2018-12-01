import json
from datetime import date, timedelta, datetime
from flask import Response
import random
# from ..core.classifier import Classifier
from flask import request
from flask import jsonify
from flask import Flask
app = Flask(__name__)

# cd ai
# FLASK_APP=hello.py flask run

@app.route("/")
def hello():
  return "Hello World!"

@app.route("/get-forcast/<province>")
def get_forcast(province):
  # return datetime.datetime.today().strftime('%Y-%m-%d')
  datas = []
  d1 = date(2018, 10, 1)  # start date
  d2 = date(2018, 12, 1)  # end dat
  delta = d2 - d1         # timedelta
  for i in range(delta.days + 1):
    datas.append({
      'created_at' : (d1 + timedelta(i)).strftime('%d-%m-%Y'),
      'hight_stream' : random.randint(50,500),
    })
  responseData = {
    'data' : datas,
    'response_code' : 200,
    'province' : province
  }
  js = json.dumps(responseData)

  resp = Response(js, status=200, mimetype='application/json')

  return resp


@app.route('/classify', methods=['POST'])
def classify():
  body = request.get_json()
  # get from body
  doc = body['doc']
  print(body['doc'])
  

  response = {
    "prediction" : doc,
    "status" :"200"
  }

  return jsonify(response)
 