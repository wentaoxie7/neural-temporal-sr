"""Evaluate configs on the held-out test sequences (full frames, run recurrently in order).

    .venv/bin/python evaluate.py                    # every config that has been trained / is a baseline
    .venv/bin/python evaluate.py configs/e1_*.json  # a subset
Writes runs/<name>/metrics.json, sample crops in results/samples/, and results/summary.md.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lu.color import domain_to_srgb, tonemap_srgb
from lu.data import build_input, frame, input_channels, load_split
from lu.metrics import delta_e76, image_analysis, psnr, ssim, temporal_flicker
from lu.models import build_model, warp

ROOT = Path(__file__).resolve().parent
import os
RUNS = Path(os.environ.get("LU_RUNS", ROOT / "runs"))
RESULTS = Path(os.environ.get("LU_RESULTS", ROOT / "results"))


def load_model(cfg):
    model = build_model(cfg, input_channels(cfg))
    if model.learned:
        ckpt = RUNS / cfg["name"] / "model.pt"
        if not ckpt.exists():
            return None
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def _batched(f):
    return {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in f.items()}


@torch.no_grad()
def evaluate_model(model, cfg, seqs, save_samples=False):
    D = cfg["domain"]
    per = defaultdict(list)
    per_preset = defaultdict(list)
    samples = {}
    for seq in seqs:
        prev_f = prev_out = prev_srgb = prev_gt = None
        for t in range(seq["lr_color"].shape[0]):
            f = _batched(frame(seq, t))
            hist = mask = None
            if cfg.get("temporal") and prev_out is not None:
                hist, mask = warp(prev_out, f["hr_mv"], prev_f["hr_depth"], f["hr_pz"])
            out = model(build_input(f, cfg, hist, mask))
            srgb = domain_to_srgb(out, D).clamp(0, 1)
            gt = tonemap_srgb(f["hr_color"] * f["exposure"].view(-1, 1, 1, 1))
            p = psnr(srgb, gt).item()
            per["psnr"].append(p)
            per["ssim"].append(ssim(srgb, gt).item())
            per["delta_e"].append(delta_e76(srgb, gt).item())
            per_preset[seq["preset"]].append(p)
            for k, v in image_analysis(srgb, gt).items():
                per[k].append(v)
            if prev_srgb is not None:
                _, m = warp(prev_srgb, f["hr_mv"], prev_f["hr_depth"], f["hr_pz"])
                ow, _ = warp(prev_srgb, f["hr_mv"])
                gw, _ = warp(prev_gt, f["hr_mv"])
                per["temporal_flicker"].append(temporal_flicker(srgb, ow, gt, gw, m))
            if save_samples and t == seq["lr_color"].shape[0] // 2:
                samples[seq["name"]] = (srgb[0], gt[0])
            prev_f, prev_out, prev_srgb, prev_gt = f, out, srgb, gt
    res = {k: float(np.mean(v)) for k, v in per.items()}
    res["psnr_by_preset"] = {k: float(np.mean(v)) for k, v in per_preset.items()}
    return res, samples


def save_sample(name, samples):
    out_dir = RESULTS / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq_name, (img, gt) in samples.items():
        a = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(a).save(out_dir / f"{seq_name}_{name}.png")
        gt_path = out_dir / f"{seq_name}_gt.png"
        if not gt_path.exists():
            Image.fromarray((gt.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(gt_path)


COLS = [("psnr", "PSNR↑", 2), ("ssim", "SSIM↑", 4), ("delta_e", "ΔE↓", 2), ("temporal_flicker", "Flicker↓", 4),
        ("sharpness_ratio", "Sharp.", 2), ("contrast_ratio", "Contr.", 2), ("mae_highlights", "MAE hi↓", 4),
        ("mae_shadows", "MAE lo↓", 4), ("mae_edges", "MAE edge↓", 4)]


def write_summary():
    rows = []
    for cfg_path in sorted((ROOT / "configs").glob("*.json")):
        cfg = json.loads(cfg_path.read_text())
        mpath = RUNS / cfg["name"] / "metrics.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        b = RUNS / cfg["name"] / "bench.json"
        bench = json.loads(b.read_text()) if b.exists() else {}
        rows.append((cfg, m, bench))
    lines = ["| Run | Params | GMACs@1080p | " + " | ".join(c[1] for c in COLS) + " | PSNR day/sunset/night |",
             "|" + "---|" * (len(COLS) + 4)]
    for cfg, m, bench in rows:
        pp = m.get("psnr_by_preset", {})
        pre = "/".join(f"{pp[k]:.1f}" if k in pp else "-" for k in ["day", "sunset", "night"])
        vals = " | ".join(f"{m[k]:.{d}f}" if k in m else "-" for k, _, d in COLS)
        params = bench.get("params", "-")
        ms = f"{bench['gmacs_1080p']:.1f}" if "gmacs_1080p" in bench else "-"
        lines.append(f"| {cfg['name']} | {params} | {ms} | {vals} | {pre} |")
    RESULTS.mkdir(exist_ok=True)
    note = ("\nSharp./Contr. are output/GT ratios (1.0 = matches reference; <1 blurrier / flatter, >1 over-sharpened). "
            "Flicker = frame-to-frame change of the output minus that of the GT, on valid reprojected pixels.\n")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n" + note)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--max-seqs", type=int, help="evaluate on only the first N test sequences (smoke tests)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    paths = [Path(p) for p in args.configs] or sorted((ROOT / "configs").glob("*.json"))
    seqs = load_split("test")[: args.max_seqs]
    for p in paths:
        cfg = json.loads(p.read_text())
        model = load_model(cfg)
        if model is None:
            print(f"[eval] {cfg['name']}: not trained yet, skipping")
            continue
        res, samples = evaluate_model(model, cfg, seqs, save_samples=True)
        (RUNS / cfg["name"]).mkdir(parents=True, exist_ok=True)
        (RUNS / cfg["name"] / "metrics.json").write_text(json.dumps(res, indent=1))
        save_sample(cfg["name"], samples)
        print(f"[eval] {cfg['name']}: PSNR {res['psnr']:.2f}  SSIM {res['ssim']:.4f}  ΔE {res['delta_e']:.2f}  "
              f"flicker {res.get('temporal_flicker', float('nan')):.4f}", flush=True)
    write_summary()


if __name__ == "__main__":
    main()
