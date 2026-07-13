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

TED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "text_encoders")
BASE_DIR = os.path.join(TED, "llama3-8b-instruct-base")
MNTP_DIR = os.path.join(TED, "McGill-NLP", "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
SUP_DIR = os.path.join(TED, "McGill-NLP", "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")


def main():
    from huggingface_hub import snapshot_download

    base_snap = snapshot_download("NousResearch/Meta-Llama-3-8B-Instruct")
    mntp_snap = snapshot_download("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
    sup_snap = snapshot_download("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")

    required = [
        (base_snap, "config.json"),
        (mntp_snap, "adapter_config.json"),
        (mntp_snap, "config.json"),
        (sup_snap, "adapter_config.json"),
    ]
    missing = [f"{directory}/{name}" for directory, name in required
               if not os.path.isfile(os.path.join(directory, name))]
    if missing:
        raise RuntimeError("downloaded encoder snapshots are incomplete: " + ", ".join(missing))

    for d in (BASE_DIR, MNTP_DIR, SUP_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    # base: symlink everything, then write a patched config.json
    for name in os.listdir(base_snap):
        if name.startswith(".") or name == "original":
            continue
        os.symlink(os.path.realpath(os.path.join(base_snap, name)), os.path.join(BASE_DIR, name))
    with open(os.path.join(base_snap, "config.json")) as fp:
        cfg = json.load(fp)
    cfg["_name_or_path"] = "meta-llama/Meta-Llama-3-8B-Instruct"
    os.remove(os.path.join(BASE_DIR, "config.json"))
    with open(os.path.join(BASE_DIR, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=1)

    # adapters: symlink files, patch adapter_config to point at the local base
    for snap, dst in ((mntp_snap, MNTP_DIR), (sup_snap, SUP_DIR)):
        for name in os.listdir(snap):
            if name.startswith("."):
                continue
            os.symlink(os.path.realpath(os.path.join(snap, name)), os.path.join(dst, name))
        ac_path = os.path.join(dst, "adapter_config.json")
        with open(ac_path) as fp:
            ac = json.load(fp)
        ac["base_model_name_or_path"] = BASE_DIR
        os.remove(ac_path)
        with open(ac_path, "w") as fp:
            json.dump(ac, fp, indent=1)

    # the MNTP repo also carries a config.json copied from the gated base; keep it
    # (it is what AutoConfig/AutoTokenizer read)
    print("TEXT_ENCODERS_DIR =", TED)
    print("run kimodo with:")
    print(f"  TEXT_ENCODERS_DIR={TED} TEXT_ENCODER_DEVICE=cpu ...")


if __name__ == "__main__":
    main()
