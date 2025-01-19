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

要进行性能评测，请运行`benchmarks/schedule_benchmarks`目录下的`benchmark.py`脚本。benchmark视频、benchmark-vip和非vip的视频请参考简介里面的展示。

Benchmark中可以配置的参数以及说明文字：
```python
# 参数区域
##########################################################
vllm_server_url = "http://localhost:15432/v1/completions"  # 本地 vLLM 推理服务地址
lambda_sentence_length = 50     # 输入promp句子长度，服从泊松分布，长度平均值（假设每个句子的单词数量）
lambda_requests = 200           # 每次并发请求的数量，服从泊松分布，数量平均值
max_sentence_length = 100       # 句子长度的最大值，默认100
lambda_request_interval = 3     # 请求间隔时间，单位：秒，同样服从泊松分布
accuracy_num = 4                # 小数点保留的位数
add_para_priority = True        # 是否给参数加上优先级
para_priority_range_min = 100    # 优先级的最小值
para_priority_range_max = 300  # 优先级的最大值
add_para_relddl = False         # 是否给参数加上相对ddl的参数
para_relddl_range_min = 1       # ddl的最小值
para_relddl_range_max = 10000   # ddl的最大值
text_display_char_num = 15      # 展示出来的输入输出字符数量
export_excel_file = False       # 是否导出Excel的表格
##########################################################
```

benchmark的数据结果在`benchmarks/schedule_benchmarks/result`的下面，具体分为日志文件和excel的表格输出。

最终我们得到的VIP和非VIP的情况如下：

### 实验一
实验配置：
- 首先我们考虑让vllm配置为priority调度策略，因为根据前面的实验效果priority性能相对较为优秀
- 非VIP的每次发送泊松分布的均值为200个请求，VIP的请求数量为40个，构成五倍的关系
- 非VIP的优先级范围控制到 $[100, 300]$，而VIP的优先级范围控制到 $[1, 150]$ ，在vllm里面优先级越小，代表的调度的越靠前
- 其余的参数保持完全一样

实验结果如下所示：

| 情况      | Total Requests | Avg Latency(s) | Avg TTFT(s) | Avg Input Len | Avg Output Len |
| --------- | :------------: | :------------: | :---------: | :-----------: | :------------: |
| VIP用户   |     640.0      |     0.6528     |   0.2852    |    50.0172    |    43.5953     |
| 非VIP用户 |     1817.0     |     2.2413     |   1.2872    |    50.2262    |    44.8277     |

解释：
- 实验时长为30s
- 我们尽可能保证每次burst并发发送出去的请求是五倍的关系，但是由于中间会sleep随机时间（服从泊松分布），所以总的请求数量可能不是严格的5倍的关系
- 最终可以看到VIP用户的请求平均延迟为0.6,但是非VIP用户的请求的延迟高达了2.24s

### 实验二：控制完全一样的burst请求数

到这里看上去好像都是没问题，但是我怀疑，很有可能这个Latency是因为模拟非VIP用户的请求时候，一次发送太多（非VIP可能一次模拟并发300个请求）导致的延迟，而并不是因为优先级调度带来的延迟。所以我设计了第二个实验。

我们控制每次burst时候发出的请求数量基本一致，然后尽可能保证vip的请求**稀疏一点**，非VIP的请求**密集一些**，看看优先级调度策略的结果：

这次实验配置：
- 还是priority调度策略，因为根据前面的实验效果priority性能相对较为优秀
- 非VIP和VIP的每次发送泊松分布的均值为100个请求
- 但是非VIP间隔设置的更短，大概1秒泊松分布均值，而VIP间隔的时间更长
- 实验输出的token和前面一样平均为50个左右


| 情况      | Total Requests | Avg Latency(s) | Avg TTFT(s) | Avg Input Len | Avg Output Len |
| --------- | :------------: | :------------: | :---------: | :-----------: | :------------: |
| VIP用户   |    828.0      |     3.1271     |   1.6215    |    50.3056    |    46.5797     |
| 非VIP用户 |    973.0      |     2.7627     |   1.2823    |    50.0524    |    44.4337     |

结果我发现这里因为VIP反而延迟更高了，我这里分析了原因，因为实验output输出token为50个，这个数量太少了！导致请求几乎可以很快的被完成。根本甚至不存在抢占的情况（或者说没有给vip用户抢占的机会！）所以我考虑增加token，设计了实验三看看瓶颈到底是不是在于token多的时候，优先级调度可以应对请求。


### 实验三：增加实验输出的token

我修改了一下的代码：
```python
output_len=random.randint(300, 500),
```

其余的参数保持还是和实验二一样，这样可以控制输出大概为400个token了，单次请求的需要推理的时间大大增加，这样就给抢占留有了足够的机会！

| 情况      | Total Requests | Avg Latency(s) | Avg TTFT(s) | Avg Input Len | Avg Output Len |
| --------- | :------------: | :------------: | :---------: | :-----------: | :------------: |
| VIP用户   |    395.0      |     12.691     |   1.2563    |    49.8076    |    334.8785    |
| 非VIP用户 |    381.0      |    16.4074     |   6.0688    |    49.6299    |    331.0761    |

这次的结果就比较符合预期了，VIP的延迟在12秒，非VIP的延迟在16秒，甚至VIP的请求数量略微的多一些，但是优先级设置的更高所以可以成功做到降低延迟。


### 实验四：换成fifo的调度
基于实验三，我们把服务端的调度策略换成fifo的策略，此外请求里面不能带优先级，否则会被服务器直接拒绝服务。看看结果，fifo在应对这种不同vip和非vip的时候效果和priority情况如何？：

| 情况      | Total Requests | Avg Latency(s) | Avg TTFT(s) | Avg Input Len | Avg Output Len |
| --------- | :------------: | :------------: | :---------: | :-----------: | :------------: |
| VIP用户   |    411.0      |    12.6767     |   3.9557    |    49.9051    |    328.326     |
| 非VIP用户 |    386.0      |    11.1861     |   1.2254    |    50.0518    |    332.5337    |

果然看到fifo情况下两个的延迟差别很小，甚至vip延迟还变高了

### 小结
- 当用户要求输出的token的数量比较小的时候，也就是推理实验很短的时候，VIP和非VIP用优先级调度区分不大
- 当用户token要求输出的很多时候，推理时间大大增加，这时候留有了充分的抢占的机会，priority调度可以很好的应对这种情况，把非vip的任务成功抢占，确保付费用户的延迟更低
- 我们还对比了fifo和priority调度，看看是否是priority调度的宫濑来保证
- 为什么不试试其他的调度策略：因为考虑到区分vip和非vip的场景，优先级是一个比较好设置的参数

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


