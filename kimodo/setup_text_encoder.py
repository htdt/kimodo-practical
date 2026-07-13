"""Assemble a local TEXT_ENCODERS_DIR for Kimodo's LLM2Vec text encoder.

The official base (meta-llama/Meta-Llama-3-8B-Instruct) is HF-gated; this
builds the same layout from the byte-identical public NousResearch mirror:

  text_encoders/
    llama3-8b-instruct-base/                       <- mirror snapshot (symlinks)
    McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/            <- MNTP adapter,
        adapter_config.base_model_name_or_path -> local base dir
    McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/ <- supervised adapter

Run kimodo with TEXT_ENCODERS_DIR=<...>/text_encoders and the wrapper joins
its hub ids onto this dir, never touching the gated repo. config.json keeps
_name_or_path = meta-llama/Meta-Llama-3-8B-Instruct so LLM2Vec's
prompt-formatting special case still fires (identical embeddings).
"""
import json
import os
import shutil

from huggingface_hub import snapshot_download

TED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "text_encoders")
BASE_DIR = os.path.join(TED, "llama3-8b-instruct-base")
MNTP_DIR = os.path.join(TED, "McGill-NLP", "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
SUP_DIR = os.path.join(TED, "McGill-NLP", "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")

base_snap = snapshot_download("NousResearch/Meta-Llama-3-8B-Instruct", local_files_only=True)
mntp_snap = snapshot_download("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", local_files_only=True)
sup_snap = snapshot_download("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", local_files_only=True)

for d in (BASE_DIR, MNTP_DIR, SUP_DIR):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)

# base: symlink everything, then write a patched config.json
for f in os.listdir(base_snap):
    if f.startswith(".") or f == "original":
        continue
    os.symlink(os.path.realpath(os.path.join(base_snap, f)), os.path.join(BASE_DIR, f))
cfg = json.load(open(os.path.join(base_snap, "config.json")))
cfg["_name_or_path"] = "meta-llama/Meta-Llama-3-8B-Instruct"
os.remove(os.path.join(BASE_DIR, "config.json"))
json.dump(cfg, open(os.path.join(BASE_DIR, "config.json"), "w"), indent=1)

# adapters: symlink files, patch adapter_config to point at the local base
for snap, dst in ((mntp_snap, MNTP_DIR), (sup_snap, SUP_DIR)):
    for f in os.listdir(snap):
        if f.startswith("."):
            continue
        os.symlink(os.path.realpath(os.path.join(snap, f)), os.path.join(dst, f))
    ac_path = os.path.join(dst, "adapter_config.json")
    ac = json.load(open(ac_path))
    ac["base_model_name_or_path"] = BASE_DIR
    os.remove(ac_path)
    json.dump(ac, open(ac_path, "w"), indent=1)

# the MNTP repo also carries a config.json copied from the gated base; keep it
# (it is what AutoConfig/AutoTokenizer read) — but verify it exists
assert os.path.exists(os.path.join(MNTP_DIR, "config.json"))
print("TEXT_ENCODERS_DIR =", TED)
print("run kimodo with:")
print(f"  TEXT_ENCODERS_DIR={TED} TEXT_ENCODER_DEVICE=cpu ...")
