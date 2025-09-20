from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
ret = generator ("In this course, we will teach you how to",
                 max_length=30,
)
print(ret)
