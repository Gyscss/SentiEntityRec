import pandas as pd
import json

# 定义文件路径
news_file_path = "../data/MINDdemo/train/news.tsv"  # 替换为你的news.tsv文件路径

# 读取news.tsv文件
column_names = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
news_data = pd.read_csv(news_file_path, sep="\t", names=column_names)

# 提取新闻标题和实体
news_titles = news_data["title"]  # 新闻标题
title_entities = news_data["title_entities"]  # 标题中的实体（JSON格式）

# 打印前5条新闻的标题和实体
for i in range(5):
    print(f"News Title {i+1}: {news_titles[i]}")
    print(f"Title Entities {i+1}: {title_entities[i]}")
    print("-" * 50)

# 解析第一条新闻的实体
example_entities = title_entities[0]
parsed_entities = json.loads(example_entities)

# 打印解析后的实体信息
print("Parsed Entities:")
for entity in parsed_entities:
    print(f"Entity: {entity['WikidataId']} - SurfaceForms:{entity['SurfaceForms']} - Confidence: {entity['Confidence']} - Label: {entity['Label']}")



