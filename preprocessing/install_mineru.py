# prereqs

sudo apt update && sudo apt install -y libglib2.0-0 libgl1 libgl1-mesa-glx

pip install --upgrade pip
pip install uv

# need permissions to modify this
sudo chmod -R 775 /opt/conda/lib/python3.11/site-packages/

# actual install
uv pip install -U "mineru[core]"
uv pip install -U "mineru[core,vllm]"

# to verify
mineru --version  # Expect: mineru 2.6.x+

# verify Python test
python -c "from mineru import MinerU; parser = MinerU(ocr_lang='eng'); print('MinerU awakened for grimoires')"

# basic usage
mineru -p <source_dir> -o <output_dir>

# Basic VLM (uses vLLM if detected)
# this may or may not work
mineru -p <source_dir> -o <output_dir> -b vlm

# Explicit vLLM Engine (Ampere-Optimized)
mineru -p <source_dir> -o <output_dir> -b vlm-vllm-engine --gpu-memory-utilization 0.9

# Multi-Thread for Throughput (If Batch-Heavy Esoterica)
mineru -p <source_dir> -o <output_dir> -b vlm-vllm-engine --data-parallel-size 1 --max-num-seqs 128

# Maybe use this 
mineru -p ./corpus_beta_plus_book_src -o ./corpus_beta_plus_book_converted/ --method auto

# USE THIS ON A4000
mineru -p ./corpus_beta_plus_book_src -o ./corpus_beta_plus_book_converted/ --method txt -b pipeline
mineru -p ./corpus_beta_plus_book_src -o ./corpus_beta_plus_book_converted/ --method txt -b vlm-transformers

# If you are having OOM issues... 
# Pre-command env (persistent in ~/.bashrc)
export VLLM_GPU_MEMORY_UTILIZATION=0.6  # Cap at 60% of 16GB (~9.6GB headroom)
export MINERU_MIN_BATCH_INFERENCE_SIZE=2  # Tiny batches for dense grimoires
export CUDA_LAUNCH_BLOCKING=1  # Sync errors for debugging (remove if slow)

# this will create a bunch of md files. use pandoc to convert
# Install once (inside your Unsloth/Paperspace container)
apt update && apt install -y pandoc

# Batch convert all .md → .txt (recursive)
find ./corpus_beta_plus_book_converted/ -name "*.md" | while read md; do
    txt="${md%.md}.txt"
    pandoc "$md" -t plain --wrap=none -o "$txt"
done