from llama_cpp import Llama

# Path to your local model gguf
model_path = "./test_models/DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf"

# Load the GGUF model directly
llm = Llama(
    model_path,
    c_ntx = 2048,
    n_threads = 8
)

prompt = (
    "I'd like you to write me a fairytale about a boy and a giant in which a great"
    "adventure occurs and a lesson is learned. Write in the style of Aesop."
)

output = llm(prompt, max_tokens=1000, echo=False)

print("Prompt:", prompt)
print("Response:", output["choices"][0]["text"].strip())
