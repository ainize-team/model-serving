import torch

from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
from typing import Dict, Tuple

# gpu 환경이 가능한지 체크
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 토크나이저를 불러오는 함수
def load_model(
    model_path: str,
) -> Tuple[AutoModelForSequenceClassification, PreTrainedTokenizerFast]:
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    return model, tokenizer


# predict 결과를 Dict로 반환해주는 함수 1 : 긍정, 0 : 부정 score는 해당 클래스일 확률 값(0~1)
@torch.no_grad()
def predict(
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
    sequence: str,
) -> Dict:
    input_vector = tokenizer.encode(sequence, return_tensors="pt").to(device)
    logit = model(input_ids=input_vector).logits
    pred = logit.argmax(dim=-1).tolist()
    if pred[0] == 1:
        return {"label": "positive", "score": logit[0][1].item()}
    else:
        return {"label": "negative", "score": logit[0][0].item()}
