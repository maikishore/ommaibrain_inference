import subprocess
import json
import os
import uuid
import re

REGISTRY_FILE = "models_registry.json"

# -----------------------------
# Registry Helpers
# -----------------------------
def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return []
    return json.load(open(REGISTRY_FILE))

def save_registry(data):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Safe App Name Generator (Modal-compatible)
# -----------------------------
def sanitize_name(name: str):
    """
    Convert model names into modal-safe names:
    - lowercase
    - replace dots with hyphens
    - keep only letters, numbers, hyphens
    """
    name = name.lower()
    name = name.replace(".", "-")
    name = re.sub(r'[^a-z0-9\-]', '-', name)
    name = re.sub(r'-+', '-', name)  # collapse repeats
    return name.strip("-")


# -----------------------------
# CLI UI Colors
# -----------------------------
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

print(f"""\n{BLUE}
=======================================================
        🚀 Modal vLLM Deployment Wizard
=======================================================
Your working base + hf-secret + safe app names + Popen
=======================================================\n{RESET}
""")

# -----------------------------
# Step 1 — Terminal Input Options
# -----------------------------
default_model = "Qwen/Qwen3-8B-FP8"
model_name = input(f"{YELLOW}Model name{RESET} [{default_model}]: ").strip() or default_model

default_revision = "main"
revision = input(f"{YELLOW}Model revision{RESET} [{default_revision}]: ").strip() or default_revision

default_served = sanitize_name(model_name.split("/")[-1])
served_name = input(f"{YELLOW}Served name{RESET} [{default_served}]: ").strip() or default_served
served_name = sanitize_name(served_name)

default_gpu = "H100"
gpu = input(f"{YELLOW}GPU type (L4 / A10G / A100 / H100){RESET} [{default_gpu}]: ").strip() or default_gpu

fast_boot_input = input(f"{YELLOW}Enable fast boot? (enforce-eager){RESET} [Y/n]: ").strip().lower()
fast_boot = fast_boot_input != "n"

# Safe app name
app_safe = sanitize_name(f"vllm-{served_name}")
deployment_id = app_safe + "-" + str(uuid.uuid4())[:6]

print(f"\n{GREEN}📦 Creating Modal script…{RESET}")


# -----------------------------
# Step 2 — Generate Modal Script
# -----------------------------
script_name = f"modal_app_{deployment_id}.py"

modal_script = f"""
import json
import subprocess
import modal
import os

# -------------------------
# Base Image
# -------------------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({{"HF_XET_HIGH_PERFORMANCE": "1"}})
)

MODEL_NAME = "{model_name}"
MODEL_REVISION = "{revision}"
FAST_BOOT = {str(fast_boot)}

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("{app_safe}")

@app.function(
    image=vllm_image,
    gpu=f"{gpu}:1",
    min_containers=1,        # replaces keep_warm
    timeout=10 * 60,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={{
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():

    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        print("🔑 HF token loaded")

    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "llm",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", "1"]

    print("Starting server with cmd:", cmd)

    subprocess.Popen(" ".join(cmd), shell=True)

if __name__ == "__main__":
    modal.run()
"""

with open(script_name, "w") as f:
    f.write(modal_script)

print(f"✔ Script created: {GREEN}{script_name}{RESET}\n")


# -----------------------------
# Step 3 — Deploy to Modal
# -----------------------------
print(f"{BLUE}🚀 Deploying to Modal…{RESET}\n")

proc = subprocess.Popen(
    ["modal", "deploy", script_name],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,   # <-- MERGE stderr (FIX)
    text=True
)

endpoint_url = None

for line in proc.stdout:
    print(line.strip())
    if "modal.run" in line:
        endpoint_url = line.strip()

proc.wait()

if not endpoint_url:
    print(f"{YELLOW}❌ Deployment failed (no endpoint found).{RESET}")
    exit()

print(f"\n{GREEN}✅ Deployment complete!{RESET}")
print(f"🔗 Endpoint: {endpoint_url}\n")

# -----------------------------
# Step 4 — Save Deployment to Registry
# -----------------------------
registry = load_registry()
registry.append({
    "name": served_name,
    "endpoint": endpoint_url,
    "model": model_name,
    "revision": revision,
    "gpu": gpu,
    "fast_boot": fast_boot
})
save_registry(registry)

print(f"{GREEN}💾 Saved to models_registry.json{RESET}")
print("Use this endpoint in Streamlit to test inference.\n")
