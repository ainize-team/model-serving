from flask import Flask, request, jsonify
import requests, json, os
from firebase_util import init_firebase, init_model
from util import *

private_key_id = os.environ['PRIVATE_KEY_ID']
private_key = os.environ['PRIVATE_KEY']

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
  with open('keys/mlops-crawler-firebase-adminsdk.json', 'r') as f:
    key = json.load(f)
    print(private_key_id)
    key['private_key_id'] = private_key_id
    key['private_key'] = private_key
    with open('keys/mlops-crawler-firebase-adminsdk.json', 'w', encoding='utf-8') as k:
      json.dump(key,k)
  init_firebase()
  init_model()
  app.run(host="0.0.0.0", port="5000")
