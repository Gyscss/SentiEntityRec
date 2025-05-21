from transformers import pipeline

# 加载 RoBERTa 情感分析模型
classifier = pipeline("sentiment-analysis", model="roberta-base")

# 定义标签映射 (根据模型的实际定义)
label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Positive',
    'LABEL_2': 'Neutral'
}

# 示例文本及实体
text = "Apple's stock has soared, and investor sentiment is high."
# text = "苹果公司股票大跌，投资者情绪低落。"
# text = "苹果公司股票平稳，投资者情绪一般。"
# text = "苹果公司产品不及预期，消费者很失望。"
entities = ["Apple"]

# 实体情感极性判断
for entity in entities:
    # 提取包含实体的上下文
    context = text  # 此处可做更精细的上下文提取

    # 使用 RoBERTa 进行情感分析
    result = classifier(context)

    # 映射标签编号到实际情感类别
    print(result)
    sentiment = label_mapping[result[0]['label']]

    # 输出情感分析结果并映射到实体
    print(f"Entity: {entity}, Sentiment: {sentiment}, Score: {result[0]['score']}")
