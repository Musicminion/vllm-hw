## 实时系统调度 vllm 大作业

### 简介
本作业修改了vllm的部分代码，用来支持四种调度策略(edf、fifo、priority、sjf)，用做了相关的性能评测。修改的代码包括：
- `benchmarks/schedule_benchmarks`：主要包括了用于评测各种调度算法用的脚本和数据集文件
- `vllm`：具体包括引擎文件、sequence Group的定义文件、core/schedule的文件等等

#### benchmark视频：

https://github.com/user-attachments/assets/da223d7d-754f-44f5-8ccf-d7e2d87ddb14

#### vip-非vip的Benchmark视频：

https://github.com/user-attachments/assets/955c7f26-4746-43f1-be49-654d4c1ac6cc




### 环境配置
创建一个conda环境：
```python
conda create -n vllm_zzq python=3.10 -y
conda activate vllm_zzq

# 不需要就退出 conda deactivate 
```

检查版本确保是`v0.6.6.post1`(一定记得得安轮子)，编译流程（这个是适合只修改python代码的）：
```bash
# 编译之前安一下轮子
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

vllm经过编译后的路径在这里（根据需要记得换一下路径）：
```
/home/zzq/.conda/envs/vllm_zzq/bin/vllm
```

如有需要启动模型前配置好代理：
```
export HTTP_PROXY=http://用户名:密码@ip:port
export HTTPS_PROXY=http://用户名:密码@ip:port
```

### 启动方法
启动一个模型：
```bash
# R760上的模型目录，视情况添加
# export HF_HOME=/home/Data/huggingface/

# 这个模型有点大加载要好久
/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-1.3b" --port 15432 --gpu_memory_utilization 0.95

# 这个小点加载快一些可以试试
/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --gpu_memory_utilization 0.95
```


### 指定调度策略
启动的时候指定调度策略(通过指定启动参数`edf`,`fifo`,`priority`,`sjf`)：

- EDF调度策略：支持用户传递参数是rel ddl，然后以这个ddl最紧迫的来进行调度
- SJF调度策略：短任务优先，一般max token越大最终的输出结果越长，推理要的时间也会越长
- Priority调度策略：优先级调度的策略
- FIFO调度策略：按照先进先出的顺序进行


例如要使用EDF调度策略，启动的时候需要加上：
```bash
# debug打开
export VLLM_LOGGING_LEVEL=DEBUG
# 四种调度策略的启动脚本
/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --gpu_memory_utilization 0.95 --scheduling_policy edf

/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --gpu_memory_utilization 0.95 --scheduling_policy sjf

/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --gpu_memory_utilization 0.95 --scheduling_policy priority

/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --gpu_memory_utilization 0.95 --scheduling_policy fcfs
```

### 发送推理请求
这里的推理请求加了`rel_deadline`，并且使用模型为opt-125m。如果不是edf调度的话不用加上这个参数（加上也不会有影响）

> 补充：这里的`rel_deadline`并不绝对的保证在这个ddl之前可以完成任务，只保证顺序排序使用。所以需要尽可能真实的设置ddl。

```
curl -X POST "http://localhost:15432/v1/completions" -H "Content-Type: application/json"     --data '{
        "model": "facebook/opt-125m",
        "prompt": "Once upon a time,",
        "max_tokens": 512,
        "temperature": 0.5,
        "rel_deadline": 100,
        "priority": 100
    }'
```



### Benchmark

要进行性能评测，请运行`benchmarks/schedule_benchmarks`目录下的`benchmark.py`脚本。

benchmark视频：

https://github.com/user-attachments/assets/d9409e8d-62e8-4011-a0eb-7fe27260b76f


VIP-非VIP的Benchmark视频：




Benchmark中可以配置的参数：
```python
# 参数区域
##########################################################
vllm_server_url = "http://localhost:15432/v1/completions"  # 本地 vLLM 推理服务地址
lambda_sentence_length = 50     # 输入promp句子长度，服从泊松分布，长度平均值（假设每个句子的单词数量）
lambda_requests = 20            # 每次并发请求的数量，服从泊松分布，数量平均值
max_sentence_length = 100       # 句子长度的最大值，默认100
lambda_request_interval = 3     # 请求间隔时间，单位：秒，同样服从泊松分布
accuracy_num = 4                # 小数点保留的位数
add_para_priority = False       # 是否给参数加上优先级
add_para_relddl = True          # 是否给参数加上相对ddl的参数
##########################################################
```

benchmark的数据结果在`benchmarks/schedule_benchmarks/result`的下面，具体分为日志文件和excel的表格输出。



### 备注
启动失败可以查看机器显存占用（模型启动失败可能是因为显存不够）
```bash
nvidia-smi
```

启动如果遇到网络问题可能是没有配置代理
```bash
export HTTP_PROXY=http://用户名:密码@ip:port
export HTTPS_PROXY=http://用户名:密码@ip:port
```

开发时vscode搜索排除建议：
```
tests/*,examples,benchmarks
```

因为我把不同调度算法排序输出的信息全都设置成debug了，所以设置vllm的日志级别：
```
export VLLM_LOGGING_LEVEL=DEBUG
```

评测需要的库：
```bash
# 进入conda环境后安装
pip install pandas pyarrow fastparquet aiohttp modelscope tabulate openpyxl
```


