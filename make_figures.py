"""Figures for the project page: temporal GIF, zoomed crops, and result charts (light + dark).

    .venv/bin/python make_figures.py        # writes assets/
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import RUNS, _batched, load_model
from lu.color import domain_to_srgb, tonemap_srgb
from lu.data import build_input, frame, load_split
from lu.models import warp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

SHOW = [("e1_bicubic", "Bicubic"), ("e1_liteunet", "U-Net (RGB)"),
        ("e4_history_tloss", "Temporal U-Net"), ("gt", "Reference (3x3 SSAA)")]

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e6e5e0",
                  s=["#2a78d6", "#eb6834", "#1baf7a"], base="#b7b5ae"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#33332f",
                 s=["#3987e5", "#d95926", "#199e70"], base="#5c5b56"),
}


def font(size):
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


@torch.no_grad()
def render_sequence(name, seq):
    """sRGB uint8 frames (H, W, 3) for one model over a whole test sequence."""
    frames = []
    if name == "gt":
        for t in range(seq["lr_color"].shape[0]):
            f = _batched(frame(seq, t))
            frames.append(tonemap_srgb(f["hr_color"] * f["exposure"].view(-1, 1, 1, 1)))
    else:
        cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        model = load_model(cfg)
        prev_f = prev_out = None
        for t in range(seq["lr_color"].shape[0]):
            f = _batched(frame(seq, t))
            hist = mask = None
            if cfg.get("temporal") and prev_out is not None:
                hist, mask = warp(prev_out, f["hr_mv"], prev_f["hr_depth"], f["hr_pz"])
            out = model(build_input(f, cfg, hist, mask))
            frames.append(domain_to_srgb(out, cfg["domain"]).clamp(0, 1))
            prev_f, prev_out = f, out
    return [(x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for x in frames]


def labelled_row(crops, labels, scale, bg=(252, 252, 251), ink=(11, 11, 11)):
    h, w = crops[0].shape[:2]
    W, H, pad, top = w * scale, h * scale, 6, 26
    img = Image.new("RGB", (len(crops) * W + (len(crops) - 1) * pad, H + top), bg)
    d = ImageDraw.Draw(img)
    f = font(15)
    for i, (c, lab) in enumerate(zip(crops, labels)):
        x = i * (W + pad)
        img.paste(Image.fromarray(c).resize((W, H), Image.NEAREST), (x, top))
        d.text((x + 2, 4), lab, fill=ink, font=f)
    return img


def make_media(seqs):
    seq = seqs[0]                                   # day scene with high-frequency ground
    vids = {n: render_sequence(n, seq) for n, _ in SHOW}
    y0, x0, h, w = 112, 224, 80, 128                # ground texture region that aliases
    labels = [lab for _, lab in SHOW]
    gif = [labelled_row([vids[n][t][y0:y0 + h, x0:x0 + w] for n, _ in SHOW], labels, 3)
           for t in range(len(vids["gt"]))]
    gif[0].save(OUT / "temporal.gif", save_all=True, append_images=gif[1:], duration=120, loop=0,
                optimize=True)

    # Full frame with the crop box, plus a static crop row from the middle of the sequence.
    t = len(vids["gt"]) // 2
    full = Image.fromarray(vids["gt"][t]).resize((384 * 2, 216 * 2), Image.LANCZOS)
    ImageDraw.Draw(full).rectangle([x0 * 2, y0 * 2, (x0 + w) * 2, (y0 + h) * 2], outline=(235, 104, 52), width=3)
    full.save(OUT / "scene.png")
    labelled_row([vids[n][t][y0:y0 + h, x0:x0 + w] for n, _ in SHOW], labels, 3).save(OUT / "crops.png")

    # Second scene (sunset), full frames side by side.
    seq2 = seqs[1]
    names2 = [("e1_bicubic", "Bicubic"), ("e1_cas", "Bicubic + CAS sharpening"),
              ("e4_history_tloss", "Temporal U-Net"), ("gt", "Reference")]
    t2 = seq2["lr_color"].shape[0] // 2
    frames2 = [render_sequence(n, seq2)[t2][60:, :] for n, _ in names2]
    labelled_row(frames2, [l for _, l in names2], 1).save(OUT / "sunset.png")


def metrics(name):
    return json.loads((RUNS / name / "metrics.json").read_text())


def style(ax, th):
    ax.set_facecolor(th["bg"])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["muted"])
    ax.tick_params(colors=th["ink2"], length=0, labelsize=10)
    ax.grid(axis="x", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def ladder_chart(th, mode):
    runs = [("e1_bicubic", "Bicubic"), ("e1_cas", "Bicubic + CAS sharpening"), ("e1_espcn", "ESPCN"),
            ("e1_liteunet", "U-Net (RGB)"), ("e3_gbuffer", "+ G-buffer"),
            ("e3_gbuffer_history", "+ motion-compensated history"), ("e4_history_tloss", "+ temporal loss")]
    m = [metrics(r) for r, _ in runs]
    labels = [l for _, l in runs]
    y = np.arange(len(runs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, facecolor=th["bg"])
    for ax, key, title, fmt, lo in [(axes[0], "psnr", "PSNR (dB), higher is better", "{:.2f}", 20),
                                    (axes[1], "temporal_flicker", "Temporal flicker, lower is better", "{:.3f}", 0)]:
        style(ax, th)
        vals = [x[key] for x in m]
        cols = [th["base"]] * 2 + [th["s"][0]] * 5
        if key == "psnr":   # dot plot: a non-zero baseline is fine for position, not for bar length
            ax.scatter(vals, y, s=90, color=cols, edgecolor=th["bg"], linewidth=2, zorder=3)
            ax.set_xlim(21.5, 27.6)
        else:
            ax.barh(y, vals, height=0.62, color=cols)
            ax.set_xlim(0, max(vals) * 1.15)
        for yi, v in zip(y, vals):
            ax.text(v + (0.15 if key == "psnr" else 0.0007), yi, fmt.format(v), va="center",
                    fontsize=9.5, color=th["ink"])
        ax.set_title(title, loc="left", color=th["ink"], fontsize=11.5, pad=8)
    axes[0].set_yticks(y, labels, color=th["ink"], fontsize=10)
    fig.text(0.01, 0.01, "Grey: classic baselines. Blue: learned models, each row adds one change. "
             "4 held-out test sequences, 96 frames.", color=th["muted"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / f"ablation_{mode}.png", dpi=170, facecolor=th["bg"])
    plt.close(fig)


def colorspace_chart(th, mode):
    runs = [("e2_linear", "Linear HDR"), ("e1_liteunet", "Mu-law"), ("e2_ldr", "Tone-mapped (LDR)")]
    bands = [("mae_shadows", "Shadows"), ("mae_midtones", "Midtones"), ("mae_highlights", "Highlights")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=th["bg"])
    style(ax, th)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=th["grid"], linewidth=0.8)
    x = np.arange(len(bands))
    bw = 0.26
    for i, (r, lab) in enumerate(runs):
        m = metrics(r)
        vals = [m[k] for k, _ in bands]
        bars = ax.bar(x + (i - 1) * (bw + 0.015), vals, bw, color=th["s"][i], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.3f}", ha="center", fontsize=8.5,
                    color=th["ink"])
    ax.set_xticks(x, [b for _, b in bands], color=th["ink"], fontsize=10.5)
    ax.set_ylabel("Mean absolute error (sRGB)", color=th["ink2"], fontsize=9.5)
    ax.set_title("Working colour space decides where the error goes", loc="left", color=th["ink"],
                 fontsize=11.5, pad=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=3)
    for t in leg.get_texts():
        t.set_color(th["ink"])
    ax.set_ylim(0, 0.095)
    fig.tight_layout()
    fig.savefig(OUT / f"colorspace_{mode}.png", dpi=170, facecolor=th["bg"])
    plt.close(fig)


def main():
    torch.set_num_threads(8)
    make_media(load_split("test"))
    for mode, th in THEMES.items():
        ladder_chart(th, mode)
        colorspace_chart(th, mode)
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
