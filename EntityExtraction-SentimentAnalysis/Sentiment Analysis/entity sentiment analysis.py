from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment(entity_text):
    inputs = tokenizer(entity_text, return_tensors="pt")
    output = model(**inputs)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    labels = ["negative", "neutral", "positive"]
    return labels[np.argmax(scores)]

text = "Tesla's stock price crashed."
sentiment = analyze_sentiment(text)

print(sentiment)  # Output: Negative
