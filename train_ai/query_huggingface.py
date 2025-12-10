#from huggingface_hub import list_models
#models = list_models(author="peers-ai")
#for model in models: print(model.id)

from huggingface_hub import list_models, list_datasets, list_spaces

author = "peers-ai"  # Replace with your HF username (or org name)

all_repos = []

# Models (public + private)
for m in list_models(author=author, limit=None):
    private_status = " (private)" if getattr(m, "private", False) else ""
    all_repos.append(("model", m.id, private_status))

# Datasets
for d in list_datasets(author=author, limit=None):
    private_status = " (private)" if getattr(d, "private", False) else ""
    all_repos.append(("dataset", d.id, private_status))

# Spaces
for s in list_spaces(author=author, limit=None):
    private_status = " (private)" if getattr(s, "private", False) else ""
    all_repos.append(("space", s.id, private_status))

# Print nicely
for repo_type, repo_id, priv in sorted(all_repos):
    print(f"{repo_type.capitalize()}: {repo_id}{priv}")
