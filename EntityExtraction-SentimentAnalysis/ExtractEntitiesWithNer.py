import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

# 这个新一些，2021年的。This model is trained on WikiNEuRal, a state-of-the-art dataset for Multilingual NER automatically derived from Wikipedia.
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# 定义函数：使用NER模型从标题中识别实体
def extract_entities_with_ner(title):
    """
    使用NER模型从新闻标题中识别实体。
    """
    ner_spans = nlp(title)
    print(ner_spans)
    ents = [span["word"] for span in ner_spans]
    print("My extracted entities:",ents)

    return ents