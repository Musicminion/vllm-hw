#!/bin/bash

echo "Running priority"

# ["fcfs", "priority", "edf", "sjf"]
/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --scheduling_policy priority > ./log.txt

