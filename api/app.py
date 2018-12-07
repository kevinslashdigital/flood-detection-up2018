import json
from ..core.classifier import Classifier
from flask import request
from flask import jsonify
from flask import Flask
app = Flask(__name__)

# cd ai
# FLASK_APP=hello.py flask run

@app.route("/")
def hello():
  return "Hello World!"

@app.route('/classify', methods=['POST'])
def classify():
  body = request.get_json()
  # get from body
  doc = body['doc']
  print(body['doc'])
  config = {
    'text_dir': 'data/dataset/doc',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/doc.model',
    'is_unicode': False
  }

  cf = Classifier(**config)
  pred = cf.classify(doc)

  response = {
    "prediction" : pred[0],
    "status" :"200"
  }

  return jsonify(response)
 