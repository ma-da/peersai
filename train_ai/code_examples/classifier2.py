from transformers import pipeline

classifier = pipeline("zero-shot-classification")
ret = classifier(
    "Vaccines are bad! Big Pharma is corrupt!",
    candidate_labels=["education", "politics", "business", "health", "medicine", "conspiracy"],
)
print(ret)
