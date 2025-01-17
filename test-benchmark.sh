#!/bin/bash

# 定义请求的 URL 和头信息
url="http://localhost:15432/v1/completions"
headers="Content-Type: application/json"

# 定义其他固定的请求数据
model="facebook/opt-125m"
prompt="Once upon a time,"
max_tokens=512
temperature=0.5

# 发送 30 个请求
for i in {1..30}
do
  # 生成一个 50 到 150 之间的随机数作为 rel_deadline
  rel_deadline=$((RANDOM % 101 + 50))

  # 构建请求数据
  data=$(cat <<EOF
{
  "model": "$model",
  "prompt": "$prompt",
  "max_tokens": $max_tokens,
  "temperature": $temperature,
  "rel_deadline": $rel_deadline
}
EOF
  )

  # 发送请求
  response=$(curl -s -X POST "$url" -H "$headers" --data "$data")
  echo "Response $i: $response"
done