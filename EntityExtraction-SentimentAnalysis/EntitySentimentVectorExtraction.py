# 要用文件存储每篇新闻的实体数据

# "news_id" "Number of entities" "Sentiment plarity&feature of entity"
#   字符串          2            {'entity1':['Wang',positive,[0.0,0.1,0.9]],}

# 用文件存储每篇新闻的实体个数，画出每篇新闻实体个数的分布图（MINDlarge和MINDsmall）
import os
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm  # 用于进度条显示
from TargetedSentimentAnalysis import target_sentiment_analysis
from ExtractEntitiesWithNer import extract_entities_with_ner

# 定义函数：解析实体并生成情感极性和情感向量（示例逻辑）
def parse_entities_news(entity_json, text):
    """
    解析实体JSON字符串，生成情感极性和情感向量（示例数据）。
    实际应用中，情感极性和向量需要通过情感分析模型生成。
    """
    if pd.isna(entity_json) or entity_json.strip() == "":  # 检查是否为空
        return {}
    entities = json.loads(entity_json)
    # print(entities[0])
    print(text)
    result = {}

    for entity in entities:
        entity_id = entity["WikidataId"]
        try:
            # 尝试执行的代码块
            entity_name = entity["SurfaceForms"][0]     # 执行这里时应该判断是否entity["SurfaceForms"]里有值；若是没值，再看entity["Label"]里是否有值；上面两个都没有值，就试着自己用NER取值；NER取不出来，就彻底没有了
            print(entity["SurfaceForms"])
        except Exception as e:
            # 如果发生指定类型的异常，则执行这里的代码
            print(f"发生了一个错误：{e}")
            result[entity_id] = [None,None,[None,None,None]]
        else:
            # 示例：生成情感极性和向量（实际需替换为情感分析模型输出）
            senti_result = target_sentiment_analysis(entity_name, text)
            # sentiment_polarity = senti_result[1]
            # sentiment_vector = senti_result[2]
            result[entity_id] = senti_result
    return result

# 定义函数：处理单个news.tsv文件
def process_news_file(news_file_path, output_file_path):
    """
    读取news.tsv文件，提取实体信息，并保存到新的TSV文件中。
    """
    # 读取news.tsv文件
    column_names = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    news_data = pd.read_csv(news_file_path, sep="\t", names=column_names)

    file_exists = os.path.exists(output_file_path)
    if file_exists:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            processed_rows = len(lines) - 1  # 减去标题行
        # 用户输入起始行（默认从已处理行继续）
        user_input = input(f"文件 {output_file_path} 已存在，已处理 {processed_rows} 行。请输入起始行号（回车默认继续）：")
        start_row = int(user_input) if user_input.strip() else processed_rows
    else:
        start_row = 0

    # 读取数据（跳过已处理的行）
    news_data = pd.read_csv(news_file_path, sep='\t', names=column_names, skiprows=start_row)

    # 打开输出文件（追加模式）
    mode = 'a' if file_exists else 'w'

    # 准备输出数据
    output_data = []
    with open(output_file_path, mode, encoding="utf-8") as f:
        # 新文件写入标题
        if not file_exists:
            f.write("news_id\tNumber of entities\tSentiment polarity&feature of entities\n")

        # 遍历每篇新闻
        for _, row in tqdm(news_data.iterrows(), total=len(news_data), desc=f"Processing {os.path.basename(news_file_path)}"):
            news_id = row["news_id"]
            print("news_id:",news_id)
            title_entities = row["title_entities"]
            abstract_entities = row["abstract_entities"]
            title = row["title"]
            abstract = row["abstract"]

            # 优先使用title_entities，如果为空则使用abstract_entities
            if not pd.isna(title_entities) and title_entities.strip() != "[]":
                # print(type(title_entities.strip()),title_entities.strip())
                # 解析实体信息
                entity_info = parse_entities_news(title_entities, title)
                print("title_entities:", title_entities)
            elif not pd.isna(abstract_entities) and abstract_entities.strip() != "[]":
                # 解析实体信息
                entity_info = parse_entities_news(abstract_entities, abstract)
                print("abstract_entities:", abstract_entities)
            else:
                entity_info = {}

            # else:
            #     entities = extract_entities_with_ner(title)
            #     title_entities = []
            #     # [{"Label": "Ukraine", "Type": "G", "WikidataId": "Q212", "Confidence": 0.946, "OccurrenceOffsets": [87], "SurfaceForms": ["Ukraine"]}]
            #     for entity in entities:
            #         title_entities.append({"Label": entity, "WikidataId":entity, "SurfaceForms": entity})
            #     title_entities = json.dumps(title_entities)
            #     # 解析实体信息
            #     entity_info = parse_entities_news(title_entities, title)

            # 将数据写入TSV文件
            f.write(f"{news_id}\t{len(entity_info)}\t{json.dumps(entity_info)}\n")

        # 添加到输出数据
        # output_data.append({
        #     "news_id": news_id,
        #     "Number of entities": len(entity_info),
        #     "Sentiment polarity&feature of entities": json.dumps(entity_info)  # 将字典转为JSON字符串
        # })

    # 保存到TSV文件
    # output_df = pd.DataFrame(output_data)
    # output_df.to_csv(output_file_path, sep="\t", index=False)

# 定义函数：处理整个MIND数据集
def process_mind_dataset(mind_root_dir, output_root_dir):
    """
    遍历MIND数据集的train/test/val文件夹，处理所有news.tsv文件。
    """
    for dataset_split in ["train","test","val"]:  # 程序结束后加入
        news_file_path = os.path.join(mind_root_dir, dataset_split, "news.tsv")
        output_file_path = os.path.join(output_root_dir, f"{dataset_split}_entities.tsv")

        if os.path.exists(news_file_path):
            print(f"Processing {dataset_split} split...")
            process_news_file(news_file_path, output_file_path)
            print(f"Saved to {output_file_path}")
        else:
            print(f"{news_file_path} does not exist. Skipping...")

# 主程序
if __name__ == "__main__":
    # 1. 获取当前时间并格式化为“年月日分时秒”
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 设置MIND数据集路径和输出路径
    # mind_large_dir = "../data/MINDlarge"  # 替换为MINDlarge路径
    # mind_small_dir = "../data/MINDsmall"  # 替换为MINDsmall路径
    mind_demo_dir = "../data/MINDdemo"  # 替换为MINDdemo路径

    # output_large_dir = f"../data/output_{current_time}/MINDlarge"  # MINDlarge输出路径
    # output_small_dir = f"../data/output_{current_time}/MINDsmall"  # MINDsmall输出路径
    # output_small_dir = f"../data/output_{current_time}/MINDdemo"  # MINDdemo输出路径

    # output_large_dir = f"../data/output/test/MINDlarge"  # MINDlarge输出路径
    # output_small_dir = f"../data/output/test/MINDsmall"  # MINDsmall输出路径
    output_demo_dir = f"../data/output/test/MINDdemo"  # MINDdemo输出路径

    # 创建输出目录
    # os.makedirs(output_large_dir, exist_ok=True)
    # os.makedirs(output_small_dir, exist_ok=True)
    os.makedirs(output_demo_dir, exist_ok=True)

    # 处理MINDlarge和MINDsmall
    # print("Processing MINDsmall...")
    # process_mind_dataset(mind_small_dir, output_small_dir)

    # print("\nProcessing MINDlarge...")
    # process_mind_dataset(mind_large_dir, output_large_dir)

    print("\nProcessing MINDdemo...")
    process_mind_dataset(mind_demo_dir, output_demo_dir)

    print("All done!")

# 实体情感极性分布图（MINDlarge和MINDsmall）

# sum(Negative)/num(entity),sum(Neutral)/num(entity),sum(Positive)/num(entity)
