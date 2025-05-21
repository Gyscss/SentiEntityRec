from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# 示例文本及实体
# text = "苹果公司股票大涨，投资者情绪高涨。"
# text = "苹果公司股票大跌，投资者情绪低落。"
# text = "苹果公司股票平稳，投资者情绪一般。"
# text = "苹果公司产品不及预期，消费者很失望。"
# text = "太糟糕了，我现在很郁闷。"
# text = "Awesome, I love it."
text = "Apple's stock has soared, and investor sentiment is high."
entities = ["Apple"]

result = classifier(text)
print(result)