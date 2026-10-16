"""End-to-end decoding throughput on one GPU: bf16 KV + FlashInfer vs exactkv.

Each request: a PL-token prompt, then G greedy tokens. The KV budget is the free memory after
loading the weights minus a headroom; each backend runs at the largest batch that fits, then
exactkv also runs at the bf16 batch size (same work, per-step time).
  python bench/e2e.py --model /path/to/Qwen2.5-7B-Instruct --out results/e2e.json
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import exactkv  # noqa: E402
from baselines import FlashInferBF16  # noqa: E402


def prompts(tok, B, T):
    """B distinct prompts of T tokens, cut from source files in site-packages."""
    import transformers
    root = os.path.dirname(transformers.__path__[0])
    files = sorted(glob.glob(root + "/transformers/models/*/modeling_*.py")) + sorted(glob.glob(root + "/torch/**/*.py", recursive=True))
    ids, text = [], ""
    for f in files:
        text += open(f, encoding="utf-8", errors="ignore").read()
        if len(text) > 400_000:
            ids += tok(text, add_special_tokens=False).input_ids
            text = ""
            if len(ids) >= B * T:
                break
    return torch.tensor(ids[: B * T], device="cuda").view(B, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--headroom-gib", type=float, default=1.6)
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    TM = a.prompt + a.gen
    fi = FlashInferBF16()
    make = {"flashinfer-bf16": lambda B: fi.engine(model, B, TM), "exactkv": lambda B: exactkv.Engine(model, B, TM)}
    torch.cuda.empty_cache()
    budget = torch.cuda.mem_get_info()[0] - a.headroom_gib * 2**30

    def per_seq(name):
        e1, e2 = make[name](1), make[name](2)
        d = e2.kv_bytes() - e1.kv_bytes()
        del e1, e2
        torch.cuda.empty_cache()
        return d
    ps = {n: per_seq(n) for n in make}
    bmax = {n: int(budget // ps[n]) for n in make}
    ids = prompts(tok, max(bmax.values()), a.prompt)

    def run(name, B):
        while B > 0:
            eng = None
            try:
                eng = make[name](B)
                torch.cuda.synchronize(); t0 = time.time()
                t = eng.prefill(ids[:B])
                torch.cuda.synchronize(); t1 = time.time()
                for _ in range(a.gen - 1):
                    t = eng.step(t).argmax(-1)
                torch.cuda.synchronize(); t2 = time.time()
                r = dict(backend=name, batch=B, kv_mib_per_seq=round(ps[name] / 2**20, 1),
                         ms_per_step=round((t2 - t1) / (a.gen - 1) * 1e3, 2),
                         decode_tokens_per_s=round(B * (a.gen - 1) / (t2 - t1), 1), prefill_s=round(t1 - t0, 1),
                         pool_grows=eng.grows)
                print(json.dumps(r), flush=True)
                return r
            except torch.cuda.OutOfMemoryError:
                print(name, "OOM at batch", B, flush=True)
                B -= 1
            finally:
                del eng
                torch.cuda.empty_cache()

    rows = [run("flashinfer-bf16", bmax["flashinfer-bf16"]), run("exactkv", bmax["exactkv"])]
    rows.append(run("exactkv", rows[0]["batch"]))
    res = dict(model=os.path.basename(a.model.rstrip("/")), gpu=torch.cuda.get_device_name(0), prompt=a.prompt,
               gen=a.gen, kv_budget_gib=round(budget / 2**30, 2), rows=rows,
               throughput_ratio=round(rows[1]["decode_tokens_per_s"] / rows[0]["decode_tokens_per_s"], 3),
               step_time_ratio_same_batch=round(rows[0]["ms_per_step"] / rows[2]["ms_per_step"], 3))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
