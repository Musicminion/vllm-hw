from dataclasses import dataclass, field
from datetime import datetime  # 确保正确导入 datetime 类
from typing import List, Optional
from tqdm.asyncio import tqdm, tqdm_asyncio
from tabulate import tabulate
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import traceback
import aiohttp
import asyncio
import random
#  __      _______ _____  
#  \ \    / /_   _|  __ \ 
#   \ \  / /  | | | |__) |
#    \ \/ /   | | |  ___/ 
#     \  /   _| |_| |     
#      \/   |_____|_|     

# 参数区域
##########################################################
vllm_server_url = "http://localhost:15432/v1/completions"  # 本地 vLLM 推理服务地址
lambda_sentence_length = 50     # 输入promp句子长度，服从泊松分布，长度平均值（假设每个句子的单词数量）
lambda_requests = 200           # 每次并发请求的数量，服从泊松分布，数量平均值
max_sentence_length = 100       # 句子长度的最大值，默认100
lambda_request_interval = 5     # 请求间隔时间，单位：秒，同样服从泊松分布
accuracy_num = 4                # 小数点保留的位数
add_para_priority = False        # 是否给参数加上优先级
para_priority_range_min = 1     # 优先级的最小值
para_priority_range_max = 10    # 优先级的最大值
add_para_relddl = False         # 是否给参数加上相对ddl的参数
para_relddl_range_min = 1       # ddl的最小值
para_relddl_range_max = 10000   # ddl的最大值
text_display_char_num = 15      # 展示出来的输入输出字符数量
export_excel_file = False       # 是否导出Excel的表格
##########################################################



# 数据准备区域
##########################################################
df = pd.read_parquet("./data/wikitext-origin/test-00000-of-00001.parquet")

# 提取文本并按句子分割
def extract_sentences(text):
    sentences = text.split(". ")  # 假设句子由句点和空格分割
    sentences = [s.strip() for s in sentences if len(s.split()) > 1]  # 过滤掉无效句子
    return sentences

sentences = []
for text in df['text']:
    sentences.extend(extract_sentences(text))

# 生成一个泊松分布的prompt的句子
def generate_sentence_from_poisson():
    sentence_length = np.random.poisson(lambda_sentence_length)
    sentence_length = min(sentence_length, max_sentence_length)  # 确保句子长度不超过最大值
    # 从数据集句子中选择一个长度合适的句子
    valid_sentences = [s for s in sentences if len(s.split()) >= sentence_length]
    selected_sentence = random.choice(valid_sentences)

    # 截取符合泊松分布长度的子句
    words = selected_sentence.split()
    truncated_sentence = " ".join(words[:sentence_length])
    return truncated_sentence
##########################################################


# 数据结构区域
##########################################################
# 参考backend_request.py里面的数据结构
# 然后定义RequestFuncInput和RequestFuncOutput两个类用来评估推理服务的性能
# 数据结构不完全一样 还是做了一定的修改的
@dataclass
class RequestFuncInput:
    batch_id: int
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    rel_deadline: Optional[float] = None
    priority: Optional[int] = None


@dataclass
class RequestFuncOutput:
    batch_id: int = 0
    generated_text: str = ""
    prompt_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    rel_deadline: Optional[float] = None
    priority: Optional[int] = None
    
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)



# 请求逻辑代码
##########################################################
# 给定的官方的评测函数，可以统计生成的时延等
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.1,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        
        if request_func_input.rel_deadline is not None:
            payload["rel_deadline"] = request_func_input.rel_deadline
        if request_func_input.priority is not None:
            payload["priority"] = request_func_input.priority        
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.prompt_text =   request_func_input.prompt
        output.rel_deadline = request_func_input.rel_deadline
        output.priority = request_func_input.priority
        output.batch_id = request_func_input.batch_id
        
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output



# 计算输出的 token的数量
def cal_token(str):
    tokens = str.split()
    return len(tokens)


# 全局变量，用于存储所有结果
results = []

async def run_single_request(if_relddl = False, if_priority = True, batch_id = 0):
    prompt_str = generate_sentence_from_poisson()
    # 创建一个 RequestFuncInput 实例
    request_input = RequestFuncInput(
        prompt=prompt_str,
        api_url=vllm_server_url,
        prompt_len=cal_token(prompt_str),
        # output_len=random.randint(10, 100),
        output_len=random.randint(300, 500),
        model="facebook/opt-125m",
        best_of=1,
        logprobs=None,
        extra_body=None,
        multi_modal_content=None,
        ignore_eos=False,
        batch_id=batch_id,
    )
    
    # 根据是否需要相对ddl和优先级设置这个参数
    if if_relddl:
        request_input.rel_deadline = random.uniform(para_relddl_range_min, para_relddl_range_max)
    if if_priority:
        request_input.priority = random.randint(para_priority_range_min, para_priority_range_max)
    
    # 创建一个 tqdm 进度条
    pbar = tqdm(total=1, disable= True)
    # 调用 async_request_openai_completions 函数
    result = await async_request_openai_completions(request_input, pbar)
    # 将结果存储到全局变量
    results.append(result)
    # 关闭进度条
    pbar.close()


def print_result():
    # 处理生成的文本，只展示前面几个单词
    def get_first_n_chars(text, n=text_display_char_num):
        # 找到换行符的位置
        newline_pos = text.find('\n')
        # 如果找到换行符且位置小于 n，则在换行符处截断
        if newline_pos != -1 and newline_pos < n:
            return text[:newline_pos] + '...'
        # 否则，按原逻辑截断
        return text[:n] + '...' if len(text) > n else text

    
    data = {
        "batch_id": [result.batch_id for result in results],
        "Promt Text": [get_first_n_chars(result.prompt_text) for result in results],
        "Generated Text": [get_first_n_chars(result.generated_text) for result in results],
        "Success": [result.success for result in results],
        "Latency": [round(result.latency, accuracy_num) for result in results],
        "TTFT": [round(result.ttft, accuracy_num) for result in results],
        # "Inter-Token Latencies": [result.itl for result in results],
        # "Avg Next-Token Latency": [result.tpot for result in results],
        "Input Len": [result.prompt_len for result in results],
        "Output Len": [cal_token(result.generated_text) for result in results],
        "Error": [result.error for result in results],
        # "Rel Deadline": [round(result.rel_deadline, accuracy_num) for result in results],  # 添加 rel_deadline
        "Rel Deadline": [round(result.rel_deadline, accuracy_num) if result.rel_deadline is not None else None for result in results],  # 添加判断
        "Priority": [result.priority for result in results]
    }
    
    df = pd.DataFrame(data)
        
    # 获取当前时间并格式化为文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{current_time}.xlsx"

    # 导出为 Excel 文件
    if export_excel_file:
        df.to_excel(filename, index=False)

    table = tabulate(df, headers='keys', tablefmt='pretty')
    tqdm.write(table)
    
    # 计算一下总请求数量，所有的Latency的平均数、TTFT的平均数、Input Len的平均数、Output Len的平均数
    # 然后作为表格输出
    total_requests = len(df)
    avg_latency = df["Latency"].mean()
    avg_ttft = df["TTFT"].mean()
    avg_input_len = df["Input Len"].mean()
    avg_output_len = df["Output Len"].mean()
    
    # 创建汇总数据框
    summary_data = {
        "Total Requests": [total_requests],
        "Avg Latency(s)": [round(avg_latency, accuracy_num)],
        "Avg TTFT(s)": [round(avg_ttft, accuracy_num)],
        "Avg Input Len": [round(avg_input_len, accuracy_num)],
        "Avg Output Len": [round(avg_output_len, accuracy_num)]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_table = tabulate(summary_df, headers='keys', tablefmt='pretty')
    tqdm.write(summary_table)
    
# 一次发起多个请求
async def run_multi_requests(num_concurrent_requests = 10, batch_id = 0):
    tasks = [run_single_request(add_para_relddl, add_para_priority, batch_id) for _ in range(num_concurrent_requests)]
    await tqdm_asyncio.gather(*tasks, desc="Processing requests")


# 模拟真实世界的请求
# 参数是模拟的时长 10s
async def simulate_real_requests(duration = 120):
    start_time = time.time()
    total_requests = 0
    cur_batchid = 0
    batchid_launchtime = []
    while time.time() - start_time < duration:
        # 模拟一个泊松分布，此次并发的次数
        parallel_nums = np.random.poisson(lambda_requests)
        total_requests += parallel_nums
        
        
        # 获取当前开始时间戳
        timestamp = time.time()
        # 将时间戳转换为可读的时间字符串
        readable_starttime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        await run_multi_requests(parallel_nums, cur_batchid)
        
        # 获取当前时间戳
        timestamp = time.time()
        # 将时间戳转换为可读的时间字符串
        readable_endtime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # 记录batchid和发起时间
        batchid_launchtime.append((cur_batchid, readable_starttime, readable_endtime, parallel_nums))
        # batchid ++
        cur_batchid = cur_batchid + 1
        # sleeptime 也服从泊松分布
        sleeptime = np.random.poisson(lambda_request_interval)
        # 即将休眠时间
        print(f"current batchid: {cur_batchid}, total requests: {total_requests}, Will sleep time: {sleeptime}")
        time.sleep(sleeptime)
    
    print("现在打印所有的batchid和发起时间：")
    # print(batchid_launchtime)
    # 将 batchid_launchtime 列表转换为 pandas 数据框
    df = pd.DataFrame(batchid_launchtime, columns=["Batch ID", "Launch Time", "End Time", "Parallel Req Nums"])

    # 使用 tabulate 库生成漂亮的表格
    table = tabulate(df, headers='keys', tablefmt='pretty')
    print(table)
    
    print("模拟结束，打印本次VIP测试的日志结果，相关文件已经保存到Excel目录下")
    print_result()
    
# 运行主函数
if __name__ == "__main__":
    asyncio.run(simulate_real_requests(30))