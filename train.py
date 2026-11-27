"""Train one experiment config on the CPU.

    .venv/bin/python train.py configs/e1_liteunet.json [--iters N] [--max-minutes M]
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from lu.data import PatchDataset, build_input, input_channels, load_split, target
from lu.losses import spatial_loss, temporal_loss
from lu.models import build_model, count_params, warp

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.iters:
        cfg["iters"] = args.iters
    if args.max_minutes:
        cfg["max_minutes"] = args.max_minutes
    run_dir = Path(args.out) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))

    torch.manual_seed(cfg.get("seed", 0))
    torch.set_num_threads(cfg.get("threads", 8))
    cin = input_channels(cfg)
    model = build_model(cfg, cin)
    if not model.learned:
        print(f"[train] {cfg['name']}: non-learned baseline, nothing to train")
        return

    seqs = load_split("train")
    iters, bs = cfg.get("iters", 4000), cfg.get("batch", 16)
    ds = PatchDataset(seqs, cfg.get("patch", 48), iters * bs)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-3), weight_decay=1e-4)
    warm = max(1, iters // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, iters) / iters)))
    weights = cfg.get("loss", {"l1": 1.0})
    w_temp = weights.get("temporal", 0.0)
    D = cfg["domain"]
    print(f"[train] {cfg['name']}: {count_params(model)} params, cin={cin}, {iters} iters x {bs}", flush=True)

    log, t0, running, it = [], time.time(), 0.0, -1
    model.train()
    for it, b in enumerate(dl):
        cur, prev = b["cur"], b["prev"]
        gt = target(cur, D)
        if cfg.get("temporal") or w_temp:
            gt_prev = target(prev, D)
            out_prev = model(build_input(prev, cfg))  # first frame of the pair: no history
            hist, mask = warp(out_prev.detach(), cur["hr_mv"], prev["hr_depth"], cur["hr_pz"])
            x_cur = build_input(cur, cfg, hist, mask) if cfg.get("temporal") else build_input(cur, cfg)
            out = model(x_cur)
            loss = spatial_loss(out, gt, weights) + 0.5 * spatial_loss(out_prev, gt_prev, weights)
            if w_temp:
                out_prev_w, _ = warp(out_prev, cur["hr_mv"])
                gt_prev_w, _ = warp(gt_prev, cur["hr_mv"])
                loss = loss + w_temp * temporal_loss(out, out_prev_w, gt, gt_prev_w, mask)
        else:
            out = model(build_input(cur, cfg))
            loss = spatial_loss(out, gt, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        running = 0.98 * running + 0.02 * loss.item() if it else loss.item()
        minutes = (time.time() - t0) / 60
        if it % 100 == 0 or it == iters - 1:
            log.append({"iter": it, "loss": running, "minutes": round(minutes, 2)})
            eta = minutes / (it + 1) * (iters - it - 1)
            print(f"[train] {cfg['name']} it {it:5d}/{iters} loss {running:.4f} "
                  f"{minutes:5.1f} min (eta {eta:.1f})", flush=True)
        if cfg.get("max_minutes") and minutes > cfg["max_minutes"]:
            print(f"[train] {cfg['name']}: time cap {cfg['max_minutes']} min reached at iter {it}")
            break

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "train_log.json").write_text(json.dumps(
        {"iters_done": it + 1, "minutes": (time.time() - t0) / 60, "params": count_params(model), "log": log}, indent=1))
    print(f"[train] {cfg['name']}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
