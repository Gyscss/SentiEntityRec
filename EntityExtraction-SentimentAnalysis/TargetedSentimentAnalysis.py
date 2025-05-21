import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from NewsSentiment import TargetSentimentClassifier
# from findAllIndexes import find_all_indexes
import numpy as np
import re
import spacy

# 加载预训练的 spaCy 语言模型（支持多语言）
nlp = spacy.load("en_core_web_sm")  # 如果是多语言，可使用 "xx_sent_ud_sm"

# 这个新一些，2021年的。This model is trained on WikiNEuRal, a state-of-the-art dataset for Multilingual NER automatically derived from Wikipedia.
# tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
# model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
# nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# TEXT = "On some great and glorious day the plain folks of the land will reach their heart's desire at last, and the White House will be adorned by a downright moron."
# TEXT = "Alibaba reported record-breaking Singles’ Day sales, exceeding analysts’ expectations. Meanwhile, Tencent announced a decline in advertising revenue due to regulatory changes. Xiaomi launched its new flagship smartphone, receiving mixed reviews from users."
# TEXT = "Alibaba reports strong Q2 growth while Alibaba faces regulatory challenges."
TEXT = "Alibaba reported record-breaking Singles’ Day sales, exceeding analysts’ expectations. Meanwhile, Tencent announced a decline in advertising revenue due to regulatory changes. Xiaomi launched its new flagship smartphone, receiving mixed reviews from users. Alibaba reports strong Q2 growth while Alibaba faces regulatory challenges."
# TEXT = "Lt. Ivan Molchanets peeked over a parapet of sand bags at the front line of the war in Ukraine. Next to him was an empty helmet propped up to trick snipers, already perforated with multiple holes."
# TEXT = "The Cost of Trump's Aid Freeze in the Trenches of Ukraine's War"

# 找到一个字符串在另一个字符串中所有出现的位置的索引列表
# def find_all_indexes(sub, s):
#     indexes = []
#     i = 0
#     while True:
#         i = s.find(sub, i)
#         if i == -1:
#             return indexes
#         indexes.append(i)
#         i += 1  # 移动到下一个字符，避免无限循环
#     return indexes

def find_all_indexes(sub,s):
    # 使用正则表达式查找所有子字符串sub在字符串s中的位置
    pattern = re.compile(re.escape(sub))
    # print('pattern:', pattern)
    matches = pattern.finditer(s)
    # print('matches:',matches,[match for match in matches])  # 返回这样的数据[<re.Match object; span=(0, 7), match='Alibaba'>, <re.Match object; span=(39, 46), match='Alibaba'>]
    return [match.start() for match in matches]

# list_indexes = find_all_indexes('Alibaba',TEXT)
# print(list_indexes)

# 定义函数：将长文本按句号分句
def split_text_into_sentences(text):
    """
    将长文本按句号分句，返回句子列表。
    """
    # text = "This is the first sentence. Here is another one! Is this the third? Yes, it is."
    # 使用 spaCy 进行分句
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    print(sentences)    # 返回这种形式的['This is the first sentence.', 'Here is another one!', 'Is this the third?', 'Yes, it is.']
    return sentences

def target_sentiment_analysis(entity,text):
    # ner_spans = nlp(TEXT)
    # print(ner_spans)

    # 根据字符串找到其在某字符串中的位置
    # ent = "Ukraine"
    # start_index = sentenct.find(entity)
    # if start_index != -1:
    #     print("子字符串的索引位置：", start_index)
    # else:
    #     print("子字符串不在主字符串中")
    # end_index = start_index + len(entity)

    if len(text) >= 100:    # 对于英文句子，Python 的 len() 函数返回的是字符数而非单词数。假设一个句子包含大约 40 个单词，每个单词平均长度大约 5 个字符，再加上单词之间的空格（通常一个空格），可以粗略估计总字符数大约为 40 × (5 + 1) = 240 个字符。
        # 解决英文文字text文字太长的问题
        sentences = split_text_into_sentences(text)
    else:
        sentences = [text]

    list_sentiment = []
    for sentenct in sentences:
        indexes = find_all_indexes(entity,sentenct)
        print(type(indexes),indexes)
        if indexes == []:
            continue
        else:
            # ner_spans = [{'entity_group': 'MISC', 'score': 0.9536458, 'word': "Ukraine", 'start': index, 'end': index+len(ent)}]

            # ents = [span["word"] for span in ner_spans]
            # print(f"Entities: {ents}")
            # tsc = TargetSentimentClassifier()

            # 定义标签映射 (根据模型的实际定义)
            # label_mapping = {
            #     'LABEL_0': 'Negative',
            #     'LABEL_1': 'Neutral',
            #     'LABEL_2': 'Positive'
            # }

            for i in indexes:
                l = sentenct[:i]
                m = sentenct[i:i+len(entity)]
                r = sentenct[i+len(entity):]
                try :
                    sentiment = TargetSentimentClassifier().infer_from_text(l, m, r)
                except Exception as e:
                    print(f"发生了一个错误：{e}")
                    continue
                # print(m, sentiment)
                # print(f"{span['entity_group']}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}\t{m}")
                # print(f"{m}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}")

                # sentiments_scores_dict = {f"{sentiment[0]['class_label']}":f"{sentiment[0]['class_prob']:.2f}",
                #                      f"{sentiment[1]['class_label']}":f"{sentiment[1]['class_prob']:.2f}",
                #                      f"{sentiment[2]['class_label']}":f"{sentiment[2]['class_prob']:.2f}"}
                # print(sentiments_scores_dict)

                sentiments_scores_list = [None, None, None]
                for dic in sentiment:
                    if dic['class_id'] == 0 :
                        sentiments_scores_list[0] = round(dic['class_prob'],4)
                    elif dic['class_id'] == 1 :
                        sentiments_scores_list[1] = round(dic['class_prob'], 4)
                    else:
                        sentiments_scores_list[2] = round(dic['class_prob'], 4)
                print(f"{m}:{sentiments_scores_list}")

                list_sentiment.append(sentiments_scores_list)

    average_values_list_sentiment = [sum(x) / len(list_sentiment) for x in zip(*list_sentiment)]  # 沿着第一个轴（列）计算平均值
    # print(average_values_list_sentiment)

    # 使用max函数找出最大值，然后使用index方法找出这个最大值的索引
    try :
        max_index = max(enumerate(average_values_list_sentiment), key=lambda x: x[1])[0]
    except Exception as e:
        print(f"发生了一个错误：{e}")
        max_index = 3
    # 得出情感极性
    if max_index == 0:
        polarity = 'nagative'
    elif max_index == 1:
        polarity = 'neutral'
    elif max_index == 2:
        polarity = 'positive'
    else:
        polarity = None

    return [entity,polarity,average_values_list_sentiment]
        # sentiment = label_mapping[result[0]['label']]
        # # 输出情感分析结果并映射到实体
        # print(f"Entity: {entity}, Sentiment: {sentiment}, Score: {result[0]['score']}")

# result = target_sentiment_analysis('Alibaba',TEXT)
# print(result)
