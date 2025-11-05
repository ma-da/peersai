from huggingface_hub import HfApi
import pandas as pd

# Initialize API with your token
# ENTER TEMPORARILY BUT DO NOT CHECK IN!
api = HfApi(token="<TOKEN_HERE>")

# List your model repositories
models = api.list_models(author="peers-ai")

# Extract relevant info
data = []
for model in models:
    data.append({
        "Model Name": model.id,
        "Created At": model.created_at,
        "Last Modified": model.last_modified,
        "Private": model.private,
        "Downloads": model.downloads,
        "Likes": model.likes,
        "URL": f"https://huggingface.co/{model.id}"
    })

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("my_huggingface_models.csv", index=False)
print("Exported to my_huggingface_models.csv")
