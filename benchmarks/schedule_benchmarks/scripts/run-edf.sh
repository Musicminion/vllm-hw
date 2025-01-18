#!/bin/bash

echo "Running edf"

/home/zzq/.conda/envs/vllm_zzq/bin/vllm serve "facebook/opt-125m" --port 15432 --scheduling_policy edf > ./log.txt

