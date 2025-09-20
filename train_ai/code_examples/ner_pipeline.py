from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ret = ner("My name is Matt and I work at Hugging Face in Brooklyn. I like pizza.")
print(ret)
