import spacy

# 加载预训练的 spaCy 语言模型（支持多语言）
nlp = spacy.load("en_core_web_sm")  # 如果是多语言，可使用 "xx_sent_ud_sm"

text = "This is the first sentence. Here is another one! Is this the third? Yes, it is."

# 使用 spaCy 进行分句
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
print(sentences)
