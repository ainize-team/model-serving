from flask import Flask, request, jsonify
import json
import os
from firebase_util import init_firebase, init_model
from util import *

private_key_id = os.environ["PRIVATE_KEY_ID"]
private_key = os.environ["PRIVATE_KEY"]
service_type = os.environ["ACCOUNT"]
project_id = os.environ["PROJECT_ID"]
client_email = os.environ["CLIENT_EMAIL"]
client_id = os.environ["CLIENT_ID"]
auth_uri = os.environ["AUTH_URI"]
token_uri = os.environ["TOKEN_URI"]
auth_provider = os.environ["AUTH_PROVIDER"]
client = os.environ["CLIENT"]

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def main():
    data = request.get_json()
    sequence = data["sequence"]
    try:
        model, tokenizer = load_model("/workspace/model")
        result = predict(model, tokenizer, sequence)
        print(result)
        return jsonify(result)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    key = {}
    key["private_key_id"] = private_key_id
    key["private_key"] = private_key
    key["type"] = service_type
    key["project_id"] = project_id
    key["client_email"] = client_email
    key["client_id"] = client_id
    key["auth_uri"] = auth_uri
    key["token_uri"] = token_uri
    key["auth_provider_x509_cert_url"] = auth_provider
    key["client_x509_cert_url"] = client

    with open("keys/mlops-crawler-firebase-adminsdk.json", "w", encoding="utf-8") as k:
        json.dump(key, k)
    init_firebase()
    init_model()
    app.run(host="0.0.0.0", port="5000")
