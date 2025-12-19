import os

# model settings
endpoint_url = "https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"

# DO NOT CHECK IN THE ACTUAL HF TOKEN
token = os.environ.get('HF_TOKEN', '')


def build_qwen_context(refs: dict, max_docs: int = 10) -> str:
    blocks = []

    for r in refs.get("results", [])[:max_docs]:
        blocks.append(
            f"[doc_id: {r['row_id']}]\n"
            f"Title: {r.get('title','')}\n"
            f"Source: {r.get('source','')}\n"
            f"Relevance: {r.get('score',0):.3f}\n\n"
            f"{r.get('text','')}".strip()
        )

    return "<EVIDENCE>\n" + "\n\n---\n\n".join(blocks) + "\n</EVIDENCE>"

def build_qwen_prompt(question: str, refs: dict) -> str:
    context = build_qwen_context(refs)

    return f"""You are a careful analyst.

Answer the question using ONLY the evidence provided.
If the evidence is insufficient, say so explicitly.
Do not use prior knowledge.
Cite sources by their doc_id in square brackets.

Question:
{question}

Evidence:
{context}

Answer:
"""

