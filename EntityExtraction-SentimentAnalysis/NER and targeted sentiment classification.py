import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from NewsSentiment import TargetSentimentClassifier

# 这个模型太老了，2018年的。仅在新闻数据集（CoNLL03）上训练
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
# nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# 这个新一些，2021年的。This model is trained on WikiNEuRal, a state-of-the-art dataset for Multilingual NER automatically derived from Wikipedia.
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# TEXT = "On some great and glorious day the plain folks of the land will reach their heart's desire at last, and the White House will be adorned by a downright moron."
# TEXT = "Alibaba reported record-breaking Singles’ Day sales, exceeding analysts’ expectations. Meanwhile, Tencent announced a decline in advertising revenue due to regulatory changes. Xiaomi launched its new flagship smartphone, receiving mixed reviews from users."
# TEXT = "Alibaba reports strong Q2 growth while Tencent faces regulatory challenges."
TEXT = "Lt. Ivan Molchanets peeked over a parapet of sand bags at the front line of the war in Ukraine. Next to him was an empty helmet propped up to trick snipers, already perforated with multiple holes."
# TEXT = "The Cost of Trump's Aid Freeze in the Trenches of Ukraine's War"

ner_spans = nlp(TEXT)
print(ner_spans)

# 根据字符串找到其在某字符串中的位置
# ent = "Ukraine"
# index = TEXT.find(ent)
# if index != -1:
#     print("子字符串的索引位置：", index)
# else:
#     print("子字符串不在主字符串中")
# ner_spans = [{'entity_group': 'MISC', 'score': 0.9536458, 'word': "Ukraine", 'start': index, 'end': index+len(ent)}]

ents = [span["word"] for span in ner_spans]
print(f"Entities: {ents}")
tsc = TargetSentimentClassifier()

# 定义标签映射 (根据模型的实际定义)
# label_mapping = {
#     'LABEL_0': 'Negative',
#     'LABEL_1': 'Neutral',
#     'LABEL_2': 'Positive'
# }

for span in ner_spans:
    l = TEXT[:span['start']]
    m = TEXT[span['start']:span['end']]
    r = TEXT[span['end']:]
    sentiment = tsc.infer_from_text(l, m, r)
    print(m, sentiment)
    # print(f"{span['entity_group']}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}\t{m}")
    print(f"{m}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}")

    sentiments_scores_dict = {f"{sentiment[0]['class_label']}":f"{sentiment[0]['class_prob']:.2f}",
                         f"{sentiment[1]['class_label']}":f"{sentiment[1]['class_prob']:.2f}",
                         f"{sentiment[2]['class_label']}":f"{sentiment[2]['class_prob']:.2f}"}
    print(sentiments_scores_dict)

    sentiments_scores_list = [None, None, None]
    for dic in sentiment:
        if dic['class_id'] == 0 :
            sentiments_scores_list[0] = round(dic['class_prob'],4)
        elif dic['class_id'] == 1 :
            sentiments_scores_list[1] = round(dic['class_prob'], 4)
        else:
            sentiments_scores_list[2] = round(dic['class_prob'], 4)
    print(f"{m}:{sentiments_scores_list}")

    # sentiment = label_mapping[result[0]['label']]
    # # 输出情感分析结果并映射到实体
    # print(f"Entity: {entity}, Sentiment: {sentiment}, Score: {result[0]['score']}")

