import json
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Response
import random
from ..core.trainer import preprocessing
from ..core.forecast import forecast
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
graph = tf.get_default_graph()
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


@app.route('/forecast', methods=['POST'])
def classify():
  body = request.get_json()
  # get from body
  # doc = body['doc']
  # print(body['doc'])
  df_sr,df_bbt,df_kc,df_ps = preprocessing("../core/dataset/flood_data.csv")
  with graph.as_default():
    result_sr = funcPrediction(df_sr,'sr')
    result_bbt = funcPrediction(df_bbt,'bt')
    result_kc = funcPrediction(df_kc,'kc')
    result_ps = funcPrediction(df_ps,'ps')
  response = {
      "sr": result_sr,
      "bbt": result_bbt,
      "kc": result_kc,
      "ps": result_ps,
      "status" :"200"
    }
  return jsonify(response)

def funcPrediction( data,name ):
    
    # get last 30 days
    last_30_days = data.tail(30)
    last_30_days = last_30_days.reset_index()
    # get last 30 days only streamHeight
    last_30_days = last_30_days['streamHeight'].values

    last_30_days_date = data.tail(30)
    last_30_days_date = last_30_days_date.reset_index()
    # get last 30 days only date
    last_30_days_date = last_30_days_date['created_at'].values
      # construct the argument parse and parse the arguments
    predictions = forecast(last_30_days,name,30, '../core/output/')
    predictions = predictions[1]
    
    # generate date to data
    last_day = data.tail(1).reset_index()['created_at'][0]
    print('last day of 30',last_day)
    start_date = last_day + timedelta(1)
    days = pd.date_range(start_date, start_date + timedelta(29), freq='D')
    new_predictions = pd.DataFrame({'day': days, 'high': predictions.flatten(), 'is_new': 1}).to_dict(orient='index')
    new_last_30_days = pd.DataFrame({'day': last_30_days_date, 'high': last_30_days.flatten(), 'is_new': 0}).to_dict(orient='index')
    list_new_predictions = []
    for key, value in new_predictions.items():
      temp = value
      list_new_predictions.append(temp)
    list_new_last_30_days = []
    for key, value in new_last_30_days.items():
      temp1 = value
      list_new_last_30_days.append(temp1)
    # merge last and predictions together
    concate_date = np.append(list_new_last_30_days, list_new_predictions)
    concate_data = {**new_last_30_days, **new_predictions}
    response = {
        "last_30_days" : last_30_days.tolist(),
        "prediction" : predictions.tolist(),
        "concate_data" : concate_date.tolist(),
      }
    return response
 