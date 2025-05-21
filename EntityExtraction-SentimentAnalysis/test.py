# from transformers import RobertaModel, RobertaTokenizer

# 加载预训练的RoBERTa模型和tokenizer
# model_name = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaModel.from_pretrained(model_name)

# 输入文本
# text = "I love China!"

# 将文本转换为token IDs
# input_ids = tokenizer.encode(text, return_tensors='pt')  #  add_special_tokens=True,

# 使用RoBERTa模型进行推理
# outputs = model(input_ids, return_dict=True)
# print(outputs)

# 得到模型的输出
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

from TargetedSentimentAnalysis import target_sentiment_analysis

# text = "Mother Nature Gives Kincade Firefighters A Reprieve; Strong Wind Storm Loses Intensity	A high wind advisory was cancelled for the North Bay hills early Wednesday, a hopeful sign that the nearly 5,000 firefighters battling the Kincade Fire can gain a stronger foothold on the blaze that has grown to 76,825 acres and destroyed 86 homes."
text = "MEMPHIS, Tenn.   A Mid-South man was arrested this week following a shooting in Southeast Memphis. According to police, a group of people were gathered at the Twin Oaks Apartments in the 3600 block of Old Street Court on October 10 when an argument broke out between Lorenzo Bailey and another man. That's when the 45-year-old suspect allegedly pulled out a gun and shot the victim. Several of ..."
length = len(text)
print(length)

entity_name = "Tenn"

senti_result = target_sentiment_analysis(entity_name, text)
print(senti_result)


