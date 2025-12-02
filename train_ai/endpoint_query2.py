from huggingface_hub import InferenceClient
import sys

# v3
endpoint_url="https://xjgkrkdcm0eqb6gr.us-east-1.aws.endpoints.huggingface.cloud"

# v4a
# endpoint_url="https://wahotlpx7kvbk9a6.us-east4.gcp.endpoints.huggingface.cloud"


token=""

print("What is your prompt?")
prompt = input()

if len(prompt) <= 1:
    print("Prompt empty so quitting.\n")
    sys.exit()

client = InferenceClient(model=endpoint_url, token=token)
response = client.text_generation(prompt, max_new_tokens=250)

print(response)
