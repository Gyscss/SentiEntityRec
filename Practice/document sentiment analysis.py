# Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("FFZG-cleopatra/Croatian-Document-News-Sentiment-Classifier-V2")

# Use a pipeline as a high-level helper
# from transformers import pipeline
# pipe = pipeline("text-classification", model="FFZG-cleopatra/Croatian-Document-News-Sentiment-Classifier")

# Use a pipeline as a high-level helper
# from transformers import pipeline
# pipe = pipeline("text-generation", model="niryuu/tinyllama-task423_persent_document_sentiment_verification-v1")

# 上面都不行

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# text = "This movie was a masterpiece! The acting, cinematography, and plot were all exceptional. Highly recommended."
text = "Alibaba reported record-breaking Singles’ Day sales, exceeding analysts’ expectations. Meanwhile, Tencent announced a decline in advertising revenue due to regulatory changes. Xiaomi launched its new flagship smartphone, receiving mixed reviews from users."

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)

# 标签：0=负面, 1=中性, 2=正面
labels = ["Negative", "Neutral", "Positive"]
predicted_label = labels[torch.argmax(predictions).item()]
print(f"Predicted Sentiment: {predicted_label}")

