from transformers import pipeline

classifier = pipeline("sentiment-analysis")
ret = classifier([
    "I've been waiting for a Hugging course my whole life.",
    "I hate this so much!"
])
print(ret)