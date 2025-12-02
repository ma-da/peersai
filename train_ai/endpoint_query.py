from huggingface_hub import InferenceClient

# v3
#endpoint_url="https://xjgkrkdcm0eqb6gr.us-east-1.aws.endpoints.huggingface.cloud"

# v4a
endpoint_url="https://j2cjmv99r8l7fpp9.us-east-1.aws.endpoints.huggingface.cloud"

# DONT CHECK IN THE TOKEN!
token=""

client = InferenceClient(model=endpoint_url, token=token)
response = client.text_generation("Who killed John F. Kennedy?", max_new_tokens=250)

print(response)
