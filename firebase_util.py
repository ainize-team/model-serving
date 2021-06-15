import firebase_admin
from firebase_admin import credentials, db, storage
import os
import json

MODEL_FILE_LIST = [
    "tokenizer.json",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "config.json",
]


def init_firebase():
    cred = credentials.Certificate("keys/mlops-crawler-firebase-adminsdk.json")
    with open("keys/firebase-config.json") as f:
        data = json.load(f)
        firebase_admin.initialize_app(cred, data)
    print("Initialize Firebase")


def init_model():
    # Asynchronous Listener
    db.reference("distribution").listen(listener)


def get_model_info():
    return db.reference("distribution").get()


def download_model(model_name, model_version):
    bucket = storage.bucket()
    if not os.path.exists("./model"):
        os.makedirs("./model")
    for file_name in MODEL_FILE_LIST:
        bucket.blob(
            f"model/{model_name}/{model_version}/{file_name}"
        ).download_to_filename(f"./model/{file_name}")
    print(f"{model_name} ver. {model_version} is downloaded.")


def listener(event):
    model_info = get_model_info()
    download_model(model_info["modelName"], model_info["modelVersion"])
