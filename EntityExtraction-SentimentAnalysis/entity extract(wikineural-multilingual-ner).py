from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 这个模型不错，还支持多语言
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
# example = "My name is Wolfgang and I live in Berlin"
# example = "Alibaba reported record-breaking Singles’ Day sales, exceeding analysts’ expectations. Meanwhile, Tencent announced a decline in advertising revenue due to regulatory changes. Xiaomi launched its new flagship smartphone, receiving mixed reviews from users."
# example = "On some great and glorious day the plain folks of the land will reach their heart's desire at last, and the White House will be adorned by a downright moron."
# example = "The Cost of Trump's Aid Freeze in the Trenches of Ukraine's War"
# example = "Lt. Ivan Molchanets peeked over a parapet of sand bags at the front line of the war in Ukraine. Next to him was an empty helmet propped up to trick snipers, already perforated with multiple holes."

# 下面的句子来自MIND-large val数据集 newsid为N29061
# example = "Info About Your Health Is More Accessible Than Ever. But Does It Help?"
example = "Improvements in medicine and technology are vital. But there can also be a downside."
# 原
ner_results = nlp(example)
# print(ner_results,type(ner_results))

# entities = [ent.text for ent in doc.ents]

# 改进
entities = [(dic['word'], dic['entity_group']) for dic in ner_results]
print(entities)  # Output: ['Tesla', 'CEO']
