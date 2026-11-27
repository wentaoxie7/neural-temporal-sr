"""Efficiency benchmark (experiment E5): params, MACs and CPU latency for a 960x540 -> 1920x1080
upscale, optional INT8 post-training quantisation, and a quality-vs-latency plot.

    .venv/bin/python benchmark.py                       # all configs with a trained model / baselines
    .venv/bin/python benchmark.py --int8 e1_liteunet    # also quantise this run and re-evaluate
"""
import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import torch

from evaluate import RESULTS, RUNS, evaluate_model, load_model, write_summary
from lu.data import PatchDataset, build_input, input_channels, load_split
from lu.models import count_macs, count_params

ROOT = Path(__file__).resolve().parent
H_LR, W_LR = 540, 960


def latency_ms(model, cin, threads, reps=10):
    torch.set_num_threads(threads)
    x = torch.rand(1, cin, H_LR, W_LR)
    with torch.no_grad():
        for _ in range(2):
            model(x)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def quantize_int8(model, cfg, cin):
    """FX-graph post-training static quantisation (per-channel weights, qnnpack/ARM backend)."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.backends.quantized.engine = "qnnpack"
    example = torch.rand(1, cin, 64, 64)
    prepared = prepare_fx(model, get_default_qconfig_mapping("qnnpack"), (example,))
    ds = PatchDataset(load_split("train"), 64, 64)
    with torch.no_grad():
        for i in range(len(ds)):
            b = {k: {kk: vv.unsqueeze(0) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
                 for k, v in ds[i].items()}
            prepared(build_input(b["cur"], cfg))
    return convert_fx(prepared)


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ms, p in rows:
        ax.scatter(ms, p)
        ax.annotate(name, (ms, p), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency, 960x540 -> 1080p (ms, log scale)")
    ax.set_ylabel("Test PSNR (dB)")
    ax.set_title("Quality vs. latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "quality_vs_latency.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", nargs="*", default=[], help="run names to quantise")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    rows = []
    for cfg_path in sorted((ROOT / "configs").glob("*.json")):
        cfg = json.loads(cfg_path.read_text())
        model = load_model(cfg)
        if model is None:
            continue
        cin = input_channels(cfg)
        x = torch.rand(1, cin, H_LR, W_LR)
        bench = {
            "params": count_params(model),
            "gmacs_1080p": count_macs(model, x) / 1e9,
            "ms_1080p": latency_ms(model, cin, args.threads),
            "ms_1080p_1thread": latency_ms(model, cin, 1, reps=3),
        }
        if cfg["name"] in args.int8:
            try:
                q = quantize_int8(model, cfg, cin)
                bench["int8_ms_1080p"] = latency_ms(q, cin, args.threads)
                bench["int8_ms_1080p_1thread"] = latency_ms(q, cin, 1, reps=3)
                torch.set_num_threads(args.threads)
                res, _ = evaluate_model(q, cfg, load_split("test"))
                bench["int8_psnr"], bench["int8_ssim"] = res["psnr"], res["ssim"]
                torch.jit.save(torch.jit.trace(q, torch.rand(1, cin, 64, 64)), RUNS / cfg["name"] / "model_int8.pt")
            except Exception as e:  # quantisation APIs vary between torch versions
                bench["int8_error"] = f"{type(e).__name__}: {e}"
        (RUNS / cfg["name"]).mkdir(parents=True, exist_ok=True)
        (RUNS / cfg["name"] / "bench.json").write_text(json.dumps(bench, indent=1))
        print(f"[bench] {cfg['name']}: {bench}", flush=True)
        m = RUNS / cfg["name"] / "metrics.json"
        if m.exists():
            rows.append((cfg["name"], bench["ms_1080p"], json.loads(m.read_text())["psnr"]))
            if "int8_psnr" in bench:
                rows.append((cfg["name"] + "-int8", bench["int8_ms_1080p"], bench["int8_psnr"]))
    RESULTS.mkdir(exist_ok=True)
    plot(rows)
    write_summary()


if __name__ == "__main__":
    main()
