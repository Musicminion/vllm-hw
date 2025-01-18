#!/bin/bash

echo "Running edf"
export VLLM_LOGGING_LEVEL=DEBUG
/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --scheduling_policy edf > ./log.txt

