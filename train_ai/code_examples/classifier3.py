from transformers import pipeline

generator = pipeline("text-generation")
ret = generator ("In this course, we will teach you how to")
print(ret)
