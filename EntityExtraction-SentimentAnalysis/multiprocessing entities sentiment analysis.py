import os
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
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
            entity_name = entity["SurfaceForms"][0]
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

    # 准备输出数据
    output_data = []
    with open(output_file_path, "w", encoding="utf-8") as f:
        # 写入表头
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

# 定义函数：处理单个数据集（train/test/val）
def process_dataset_split(args):
    """
    处理单个数据集（train/test/val）。
    """
    news_file_path, output_file_path = args
    if os.path.exists(news_file_path):
        print(f"Processing {os.path.basename(news_file_path)}...")
        process_news_file(news_file_path, output_file_path)
        print(f"Saved to {output_file_path}")
    else:
        print(f"{news_file_path} does not exist. Skipping...")

# 主程序
if __name__ == "__main__":
    # 设置MIND数据集路径和输出路径
    mind_large_dir = "../data/MINDlarge"  # 替换为MINDlarge路径
    mind_small_dir = "../data/MINDsmall"  # 替换为MINDsmall路径
    output_large_dir = "../data/output/MINDlarge"  # MINDlarge输出路径
    output_small_dir = "../data/output/MINDsmall"  # MINDsmall输出路径

    # 创建输出目录
    os.makedirs(output_large_dir, exist_ok=True)
    os.makedirs(output_small_dir, exist_ok=True)

    # 准备任务列表
    tasks = []
    for dataset_dir, output_dir in [(mind_small_dir, output_small_dir), (mind_large_dir, output_large_dir)]:
        for dataset_split in ["train", "test", "val"]:
            news_file_path = os.path.join(dataset_dir, dataset_split, "news.tsv")
            output_file_path = os.path.join(output_dir, f"{dataset_split}_entities.tsv")
            tasks.append((news_file_path, output_file_path))
        print(dataset_dir, dataset_split)

    # 使用多进程并行处理
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_dataset_split, tasks)

    print("All done!")