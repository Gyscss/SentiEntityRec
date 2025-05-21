from transformers import pipeline

# 直接使用Hugging Face Pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# # 输入句子
# sentences = [
#     "This movie is absolutely fantastic!",
#     "The service was slow and unfriendly."
# ]
#
# # 批量预测
# results = classifier(sentences)
# for sentence, result in zip(sentences, results):
#     print(f"Sentence: {sentence}\nLabel: {result['label']}, Score: {result['score']:.2f}\n")


sentences = "Alibaba and Tencent released financial reports, but Tencent's advertising revenue declined, while Alibaba's e-commerce growth remained strong."
result = classifier(sentences)
print(result)

