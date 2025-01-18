#!/bin/bash

echo "Running fcfs"
export VLLM_LOGGING_LEVEL=DEBUG
# ["fcfs", "priority", "edf", "sjf"]
/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --scheduling_policy fcfs > ./log.txt

