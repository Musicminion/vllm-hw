import numpy as np
import random
import time
import requests
import json
import pandas as pd

# 设置泊松分布的参数
lambda_requests = 800  # 每分钟的平均请求频率
lambda_sentence_length = 50  # 句子长度的平均值（假设每个句子的单词数量）
max_sentence_length = 100  # 句子长度的最大值
request_interval = 1  # 请求间隔时间，单位：秒
vllm_server_url = "http://localhost:15432/v1/completions"  # 本地 vLLM 推理服务地址

# 读取 Parquet 文件
df = pd.read_parquet("./data/wikitext-origin/test-00000-of-00001.parquet")


# 提取文本并按句子分割
def extract_sentences(text):
    sentences = text.split(". ")  # 假设句子由句点和空格分割
    sentences = [s.strip() for s in sentences if len(s.split()) > 1]  # 过滤掉无效句子
    return sentences

sentences = []
for text in df['text']:
    sentences.extend(extract_sentences(text))
    
# 然后遍历打印输出sentences
