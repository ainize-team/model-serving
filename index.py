from flask import Flask, request, jsonify
import requests
import json
from firebase_util import init_firebase, init_model
from util import *

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def main():
  data = request.get_json()
  sequence = data['sequence']
  try:
    model, tokenizer = load_model('/workspace/model')
    result = predict(model, tokenizer, sequence)
    print(result)
    return jsonify(result)
  except Exception as e:
    print(e)

if __name__ == '__main__':
  init_firebase()
  init_model()
  app.run(host="0.0.0.0", port="5000")
