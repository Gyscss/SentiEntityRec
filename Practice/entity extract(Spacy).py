import spacy
nlp = spacy.load("en_core_web_sm")
# headline = "Tesla's stock price crashed after CEO's announcement."
# headline = "Opinion: NFL had no choice but to send a clear message with Garrett punishment"
# headline = "Apple's stock has soared, and investor sentiment is high."
# headline = "Apple releases new iPhone to positive reviews"
# headline = "Wang Qingshuai is a good person."
# headline = "Alibaba reported record-breaking Singles’ Day sales, exceeding analysts’ expectations. Meanwhile, Tencent announced a decline in advertising revenue due to regulatory changes. Xiaomi launched its new flagship smartphone, receiving mixed reviews from users."
headline = "Opinion: NFL had no choice but to send a clear message with Garrett punishment"
doc = nlp(headline)
# 原
# entities = [ent.text for ent in doc.ents]

# 改进
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(entities)  # Output: ['Tesla', 'CEO']
