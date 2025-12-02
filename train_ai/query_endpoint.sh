#!/bin/bash

# Replace with your actual endpoint URL
ENDPOINT="https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"

# DONT CHECK IN THE TOKEN!
HF_TOKEN=""

#echo "Querying endpoint $ENDPOINT"

echo "Checking health..."
curl -s $ENDPOINT/healthz

echo -e "\n\nChecking ready..."
curl -s $ENDPOINT/ready

echo -e "\n\nTiny test request..."
curl -s $ENDPOINT/generate \
  -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"inputs":"[warmup]","parameters":{"max_new_tokens":8}}' | jq -r .generated_text
