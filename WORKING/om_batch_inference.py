# ======================================================
# big_model_batch_runner.py
# For large models (≥30B) using multi-GPU tensor parallelism.
# Optimized for H100/A100/A10G multi-GPU clusters.
# ======================================================

import os
import json
import modal
import subprocess
import re
from typing import List, Dict
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"    # Example 32B model
MODEL_REVISION = "main"

# Choose GPU layout (Modal nodes)
GPU_LAYOUT = "A10G:4"      # 4×A10G
# Options:
#   "A10G:4"   → fits 32B models
#   "A10G:8"   → fits 70B models
#   "H100:1"   → fits 30–70B in FP8
#   "H100:2"   → comfortable for 70B+
#   "A100:4"   → fits 30–70B

# Chunk size for batch generation
DEFAULT_CHUNK_SIZE = 8

# Timeout increased for large models
BUILD_TIMEOUT = 1200   # 20 minutes


app = modal.App("big-vllm-batch-runner")

# ======================================================
# Multi-GPU vLLM image
# ======================================================

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2"
    )
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ======================================================
# HIGH-CAPACITY AUTO-TUNER
# ======================================================

def detect_total_gpus(layout: str) -> int:
    """Parse 'A10G:4' => 4"""
    if ":" in layout:
        return int(layout.split(":")[1])
    return 1

def big_model_auto_tune(model_name: str, gpu_layout: str) -> Dict:
    """Auto-tune tensor_parallel_size, memory usage, sequencing."""
    total_gpus = detect_total_gpus(gpu_layout)
    size = int(re.search(r"(\d+)[Bb]", model_name).group(1))

    print(f"🧠 BIG-AutoTuner: Model={size}B, GPUs={total_gpus}")

    config = {
        "tensor_parallel_size": total_gpus,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 1024,
        "max_num_seqs": 1
    }

    # Special settings for 70B
    if size >= 60:
        config["gpu_memory_utilization"] = 0.85
        config["max_model_len"] = 768
        config["max_num_seqs"] = 1

    return config


# ======================================================
# MULTI-GPU WORKER
# ======================================================

@app.function(
    image=vllm_image,
    gpu=GPU_LAYOUT,
    timeout=BUILD_TIMEOUT,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    }
)
def big_batch_worker(prompts: List[str]) -> List[str]:
    """Executes vLLM on multi-GPU for huge models."""
    from vllm import LLM, SamplingParams

    # Token for private models
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    if not hasattr(big_batch_worker, "llm"):
        print(f"🔥 Loading LARGE model {MODEL_NAME}... (multi-GPU)")
        tuned = big_model_auto_tune(MODEL_NAME, GPU_LAYOUT)

        big_batch_worker.llm = LLLM = LLM(
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            tensor_parallel_size=tuned["tensor_parallel_size"],
            gpu_memory_utilization=tuned["gpu_memory_utilization"],
            max_model_len=tuned["max_model_len"],
            max_num_seqs=tuned["max_num_seqs"]
        )

        big_batch_worker.params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=256
        )

    print(f"⚡ Running batch of {len(prompts)} on {GPU_LAYOUT}...")

    outputs = big_batch_worker.llm.generate(prompts, big_batch_worker.params)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


# ======================================================
# BATCH RUNNER
# ======================================================

class BigBatchRunner:
    def __init__(self, dataset: List[str], chunk_size=DEFAULT_CHUNK_SIZE):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.results = []

    def chunk_dataset(self):
        chunks = [
            self.dataset[i:i + self.chunk_size]
            for i in range(0, len(self.dataset), self.chunk_size)
        ]
        print(f"📦 Total chunks: {len(chunks)} (chunk_size={self.chunk_size})")
        return chunks

    def run_remote(self, chunks):
        all_outputs = []
        print(f"🚀 Dispatching {len(chunks)} jobs to multi-GPU cluster {GPU_LAYOUT}...")

        for batch in tqdm(big_batch_worker.map(chunks), total=len(chunks)):
            all_outputs.extend(batch)

        self.results = all_outputs
        return all_outputs

    def save_results(self, out="big_outputs.json"):
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Saved results to {out}")


# ======================================================
# ENTRYPOINT
# ======================================================

@app.local_entrypoint()
def run(input_file="prompts.json", output_file="big_outputs.json", chunk=DEFAULT_CHUNK_SIZE):
    print(f"📥 Loading {input_file}...")
    dataset = json.load(open(input_file))

    runner = BigBatchRunner(dataset, chunk_size=chunk)
    chunks = runner.chunk_dataset()
    runner.run_remote(chunks)
    runner.save_results(output_file)
