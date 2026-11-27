"""Dataset generation, loading and network-input assembly."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .color import to_domain
from .render import render_sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SPLITS = {"train": (1000, 20), "test": (9000, 4)}  # (seed offset, number of sequences)
FRAMES = 24


def generate(frames: int = FRAMES, splits=SPLITS, out_dir: Path = DATA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    t0 = time.time()
    for split, (offset, n) in splits.items():
        for k in range(n):
            path = out_dir / f"{split}_{k:03d}.npz"
            if path.exists():
                continue
            seq = render_sequence(offset + k, frames)
            np.savez(path, **seq)
            meta[path.name] = {"preset": str(seq["preset"]), "exposure": float(seq["exposure"])}
            print(f"[data] {path.name} {seq['preset']} ({time.time() - t0:.0f}s)", flush=True)
    if meta:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def load_split(split: str, data_dir: Path = DATA_DIR):
    files = sorted(data_dir.glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No {split} data in {data_dir}; run `python generate_data.py` first")
    seqs = []
    for f in files:
        z = np.load(f)
        seq = {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files if k != "preset"}
        seq["name"] = f.stem
        seq["preset"] = str(z["preset"])
        seqs.append(seq)
    return seqs


def frame(seq, t, crop=None):
    """Slice frame t of a sequence (optionally an LR crop (y, x, size) and matching HR crop)."""
    d = {}
    for k, v in seq.items():
        if not isinstance(v, torch.Tensor) or v.ndim < 2:
            continue
        x = v[t]
        if crop is not None and x.ndim == 3:
            y0, x0, s = crop
            if k.startswith("lr_"):
                x = x[:, y0:y0 + s, x0:x0 + s]
            elif k.startswith("hr_"):
                x = x[:, 2 * y0:2 * y0 + 2 * s, 2 * x0:2 * x0 + 2 * s]
        d[k] = x
    d["exposure"] = seq["exposure"]
    return d


class PatchDataset(torch.utils.data.Dataset):
    """Random LR patches with their HR targets, plus the previous frame for temporal training."""

    def __init__(self, seqs, patch: int, length: int):
        self.seqs, self.patch, self.length = seqs, patch, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i * 7919 + int(torch.initial_seed() % 100000))
        seq = self.seqs[torch.randint(len(self.seqs), (1,), generator=g).item()]
        T, _, H, W = seq["lr_color"].shape
        t = torch.randint(1, T, (1,), generator=g).item()
        y0 = torch.randint(0, H - self.patch + 1, (1,), generator=g).item()
        x0 = torch.randint(0, W - self.patch + 1, (1,), generator=g).item()
        crop = (y0, x0, self.patch)
        cur, prev = frame(seq, t, crop), frame(seq, t - 1, crop)
        return {"cur": cur, "prev": prev}


def batch_to(b, dtype=torch.float32):
    return {k: (v.to(dtype) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


# ---------------------------------------------------------------------------- input assembly
def _exp(f):
    e = f["exposure"]
    return e.view(-1, 1, 1, 1) if e.ndim >= 1 else e


def input_channels(cfg) -> int:
    c = 3
    if cfg.get("gbuffer"):
        c += 3 + 3 + 1 + 2      # normal, albedo, depth, jitter
    if cfg.get("temporal"):
        c += 12 + 1             # space-to-depth warped history + validity mask
    return c


def target(f, domain):
    return to_domain(f["hr_color"] * _exp(f), domain)


def lr_color(f, domain):
    return to_domain(f["lr_color"] * _exp(f), domain)


def build_input(f, cfg, history=None, hist_mask=None):
    """Assemble network input at LR resolution. `history`: warped previous HR output (domain)."""
    x = [lr_color(f, cfg["domain"])]
    if cfg.get("gbuffer"):
        N, _, H, W = x[0].shape
        depth = torch.log1p(f["lr_depth"].clamp(min=0, max=1e3)) / np.log1p(1e3)
        jit = f["jitter"].view(N, 2, 1, 1).expand(N, 2, H, W) - 0.5
        x += [f["lr_normal"], f["lr_albedo"], depth, jit]
    if cfg.get("temporal"):
        N, _, H, W = x[0].shape
        if history is None:
            history = torch.zeros(N, 3, 2 * H, 2 * W)
            hist_mask = torch.zeros(N, 1, 2 * H, 2 * W)
        x += [F.pixel_unshuffle(history * hist_mask, 2), F.avg_pool2d(hist_mask, 2)]
    return torch.cat(x, 1)
