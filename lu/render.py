"""A tiny vectorised ray-caster that produces game-like training data on the CPU.

For every frame of a camera fly-through it renders, like a game engine feeding a temporal upscaler:
  LR input : 1 sample/pixel at a per-frame sub-pixel jitter (Halton 2,3), linear HDR radiance
             plus G-buffer (world normal, albedo, linear depth) and motion vectors
  HR target: 3x3 supersampled (anti-aliased) linear HDR radiance at 2x resolution
  both     : motion vectors to the previous frame and the expected depth in the previous camera,
             used for warping and disocclusion masks.

Scenes: textured ground plane (high-frequency procedural textures that alias when minified),
spheres with striped/checker/solid albedo, Blinn-Phong specular, emissive spheres that act as
point lights (HDR highlights), a sun with hard shadows and an HDR sky with a sun disc.
Lighting presets cover day, sunset and night, so dynamic range varies a lot between scenes.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

SKY_DEPTH = 1e4


def halton(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


@dataclass
class Camera:
    pos: torch.Tensor
    fwd: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    tan_half: float
    aspect: float

    @staticmethod
    def look_at(pos, target, fov_deg=60.0, aspect=16 / 9):
        pos = torch.as_tensor(pos, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        fwd = _normalize(target - pos)
        right = _normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0])))
        up = torch.linalg.cross(right, fwd)
        return Camera(pos, fwd, right, up, math.tan(math.radians(fov_deg) / 2), aspect)

    def rays(self, H, W, ox, oy):
        """Ray directions for pixel coordinates (j+ox, i+oy); ox/oy scalars or (H,W) tensors."""
        j = torch.arange(W, dtype=torch.float32).view(1, W)
        i = torch.arange(H, dtype=torch.float32).view(H, 1)
        x = ((j + ox) / W * 2 - 1) * self.tan_half * self.aspect
        y = (1 - (i + oy) / H * 2) * self.tan_half
        x, y = torch.broadcast_tensors(x, y)
        d = self.fwd + x[..., None] * self.right + y[..., None] * self.up
        return _normalize(d)

    def project(self, p, H, W):
        """World points (...,3) -> continuous pixel coords (u, v) and view depth z."""
        v = p - self.pos
        z = (v * self.fwd).sum(-1)
        zc = z.clamp(min=1e-4)
        x = (v * self.right).sum(-1) / (zc * self.tan_half * self.aspect)
        y = (v * self.up).sum(-1) / (zc * self.tan_half)
        return (x + 1) / 2 * W, (1 - y) / 2 * H, z


class Scene:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        preset = rng.choice(["day", "sunset", "night"], p=[0.45, 0.3, 0.25])
        self.preset = preset
        elev = {"day": rng.uniform(35, 70), "sunset": rng.uniform(4, 15), "night": rng.uniform(20, 50)}[preset]
        azim = rng.uniform(0, 360)
        e, a = math.radians(elev), math.radians(azim)
        self.sun_dir = torch.tensor([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)], dtype=torch.float32)
        if preset == "day":
            self.sun_col = torch.tensor([1.0, 0.97, 0.9]) * rng.uniform(3, 6)
            self.sky_zen = torch.tensor([0.25, 0.45, 0.9]) * rng.uniform(0.8, 1.4)
            self.sky_hor = torch.tensor([0.75, 0.85, 1.0]) * rng.uniform(0.8, 1.4)
        elif preset == "sunset":
            self.sun_col = torch.tensor([1.0, 0.55, 0.25]) * rng.uniform(2, 4)
            self.sky_zen = torch.tensor([0.15, 0.2, 0.45]) * rng.uniform(0.5, 1.0)
            self.sky_hor = torch.tensor([1.0, 0.5, 0.25]) * rng.uniform(0.8, 1.5)
        else:  # moonlight + strong local emitters
            self.sun_col = torch.tensor([0.6, 0.7, 1.0]) * rng.uniform(0.05, 0.15)
            self.sky_zen = torch.tensor([0.01, 0.015, 0.04])
            self.sky_hor = torch.tensor([0.03, 0.035, 0.06])
        self.sun_disc = self.sun_col * (40.0 if preset != "night" else 150.0)

        # ground texture parameters
        self.g_cols = torch.tensor(rng.uniform(0.05, 0.85, (3, 3)), dtype=torch.float32)
        self.g_type = int(rng.integers(0, 4))
        self.g_scale = float(rng.uniform(0.4, 1.5))
        self.g_fine = float(rng.uniform(6, 14))

        # spheres: centre, radius, albedo a/b, pattern, frequency, specular, shininess, emission
        n = int(rng.integers(6, 12))
        n_emit = int(rng.integers(2, 5)) if preset == "night" else int(rng.integers(0, 2))
        cs, rs, ca, cb, pat, freq, spec, shin, emit = [], [], [], [], [], [], [], [], []
        for k in range(n):
            r = float(rng.uniform(0.3, 1.4)) if k >= n_emit else float(rng.uniform(0.15, 0.4))
            x, z = rng.uniform(-8, 8), rng.uniform(-8, 8)
            y = r if rng.random() < 0.75 or k < n_emit else r + rng.uniform(0.3, 2.0)
            cs.append([x, y, z]); rs.append(r)
            ca.append(rng.uniform(0.05, 0.95, 3)); cb.append(rng.uniform(0.05, 0.95, 3))
            pat.append(int(rng.integers(0, 3))); freq.append(float(rng.uniform(4, 24)))
            spec.append(float(rng.uniform(0.0, 0.6))); shin.append(float(rng.uniform(8, 128)))
            if k < n_emit:
                col = rng.uniform(0.3, 1.0, 3); col = col / col.max()
                emit.append(col * rng.uniform(8, 30))
            else:
                emit.append([0.0, 0.0, 0.0])
        t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
        self.sc, self.sr, self.sa, self.sb = t(cs), t(rs), t(ca), t(cb)
        self.spat, self.sfreq, self.sspec, self.sshin, self.semit = t(pat), t(freq), t(spec), t(shin), t(emit)
        self.n_emit = n_emit

    # -- geometry ---------------------------------------------------------------------------
    def _spheres_t(self, o, d):
        """Nearest positive hit distance per sphere: (N, S)."""
        oc = o[:, None, :] - self.sc[None]                 # N,S,3
        b = (oc * d[:, None, :]).sum(-1)
        c = (oc * oc).sum(-1) - self.sr[None] ** 2
        disc = b * b - c
        sq = disc.clamp(min=0).sqrt()
        t0, t1 = -b - sq, -b + sq
        t = torch.where(t0 > 1e-3, t0, t1)
        return torch.where((disc > 0) & (t > 1e-3), t, torch.full_like(t, float("inf")))

    def intersect(self, o, d):
        ts = self._spheres_t(o, d)
        t_s, idx = ts.min(-1)
        t_p = torch.where(d[:, 1] < -1e-6, -o[:, 1] / d[:, 1].clamp(max=-1e-6), torch.full_like(t_s, float("inf")))
        t_p = torch.where(t_p > 1e-3, t_p, torch.full_like(t_p, float("inf")))
        is_sphere = t_s < t_p
        t = torch.minimum(t_s, t_p)
        hit = torch.isfinite(t)
        return t, hit, is_sphere & hit, idx

    # -- shading ----------------------------------------------------------------------------
    def _ground_albedo(self, p):
        x, z = p[:, 0] / self.g_scale, p[:, 2] / self.g_scale
        c0, c1, c2 = self.g_cols
        checker = ((torch.floor(x) + torch.floor(z)) % 2)[:, None]
        if self.g_type == 0:      # checkerboard + fine stripes
            m = checker
            fine = 0.5 + 0.5 * torch.sign(torch.sin(2 * math.pi * self.g_fine * x))
        elif self.g_type == 1:    # tiles with thin grout lines
            fx, fz = x - torch.floor(x), z - torch.floor(z)
            m = ((fx < 0.06) | (fz < 0.06)).float()[:, None]
            fine = 0.5 + 0.5 * torch.sin(2 * math.pi * self.g_fine * (x + z))
        elif self.g_type == 2:    # concentric rings (moire-prone)
            m = (0.5 + 0.5 * torch.sign(torch.sin(2 * math.pi * 1.5 * torch.sqrt(x * x + z * z))))[:, None]
            fine = 0.5 + 0.5 * torch.sign(torch.sin(2 * math.pi * self.g_fine * z))
        else:                     # planks with high-frequency grain
            m = ((torch.floor(z * 2) % 2))[:, None]
            fine = 0.5 + 0.5 * torch.sin(2 * math.pi * self.g_fine * 3 * x + 3 * torch.sin(z * 7))
        base = c0 * (1 - m) + c1 * m
        return base * (0.75 + 0.25 * fine[:, None]) + 0.05 * c2

    def _sphere_albedo(self, idx, n):
        u = torch.atan2(n[:, 2], n[:, 0]) / (2 * math.pi) + 0.5
        v = torch.acos(n[:, 1].clamp(-1, 1)) / math.pi
        f = self.sfreq[idx]
        pat = self.spat[idx]
        stripes = 0.5 + 0.5 * torch.sign(torch.sin(2 * math.pi * f * v))
        checker = ((torch.floor(u * f * 2) + torch.floor(v * f)) % 2)
        m = torch.where(pat == 0, stripes, torch.where(pat == 1, checker, torch.zeros_like(u)))[:, None]
        return self.sa[idx] * (1 - m) + self.sb[idx] * m

    def sky(self, d):
        h = d[:, 1:2].clamp(min=0) ** 0.5
        col = self.sky_hor * (1 - h) + self.sky_zen * h
        below = (d[:, 1:2] < 0).float()
        col = col * (1 - below) + self.sky_hor * 0.3 * below
        disc = ((d * self.sun_dir).sum(-1, keepdim=True) > 0.9995).float()
        return col + disc * self.sun_disc

    def shade(self, o, d):
        """Returns radiance, normal, albedo, world position, depth-along-ray, hit mask."""
        N = o.shape[0]
        t, hit, is_s, idx = self.intersect(o, d)
        p = o + d * torch.where(hit, t, torch.zeros_like(t))[:, None]
        n = torch.zeros(N, 3)
        alb = torch.zeros(N, 3)
        rad = self.sky(d)
        if hit.any():
            n_s = _normalize(p - self.sc[idx])
            n = torch.where(is_s[:, None], n_s, torch.tensor([0.0, 1.0, 0.0]).expand(N, 3))
            alb_s = self._sphere_albedo(idx, n_s)
            alb = torch.where(is_s[:, None], alb_s, self._ground_albedo(p))
            alb = torch.where(hit[:, None], alb, torch.zeros_like(alb))
            # sun + hard shadow
            so = p + n * 2e-3
            sd = self.sun_dir.expand(N, 3)
            vis = (~torch.isfinite(self._spheres_t(so, sd).min(-1).values)).float()[:, None]
            ndl = (n * sd).sum(-1, keepdim=True).clamp(min=0)
            amb = (self.sky_zen * 0.6 + self.sky_hor * 0.4) * (0.55 + 0.45 * n[:, 1:2])
            col = alb * (self.sun_col * ndl * vis + amb)
            # specular
            h = _normalize(sd - d)
            spec_k = torch.where(is_s, self.sspec[idx], torch.full_like(t, 0.08))[:, None]
            shin = torch.where(is_s, self.sshin[idx], torch.full_like(t, 32.0))[:, None]
            col = col + spec_k * self.sun_col * vis * (n * h).sum(-1, keepdim=True).clamp(min=0) ** shin
            # emissive spheres as unshadowed point lights
            for k in range(self.n_emit):
                lv = self.sc[k] - p
                dist2 = (lv * lv).sum(-1, keepdim=True).clamp(min=1e-2)
                ldir = lv / dist2.sqrt()
                ndl_k = (n * ldir).sum(-1, keepdim=True).clamp(min=0)
                col = col + alb * self.semit[k] * (self.sr[k] ** 2) * ndl_k / dist2
            emit = torch.where(is_s[:, None], self.semit[idx], torch.zeros(N, 3))
            col = torch.where(is_s[:, None] & (emit.sum(-1, keepdim=True) > 0), emit, col)
            rad = torch.where(hit[:, None], col, rad)
        depth = torch.where(hit, t, torch.full_like(t, SKY_DEPTH))
        p = torch.where(hit[:, None], p, o + d * SKY_DEPTH)
        return rad, n, alb, p, depth, hit


def camera_path(rng: np.random.Generator, n_frames: int):
    """Smooth fly-through: orbit, strafe or dolly, with mild bobbing."""
    kind = rng.choice(["orbit", "strafe", "dolly"])
    radius, height = rng.uniform(7, 12), rng.uniform(1.2, 4.0)
    a0, speed = rng.uniform(0, 2 * math.pi), rng.uniform(0.015, 0.035) * rng.choice([-1, 1])
    target0 = np.array([rng.uniform(-2, 2), rng.uniform(0.3, 1.0), rng.uniform(-2, 2)])
    cams = []
    for f in range(n_frames):
        if kind == "orbit":
            a = a0 + speed * f
            pos = [radius * math.cos(a), height + 0.15 * math.sin(f * 0.3), radius * math.sin(a)]
            tgt = target0
        elif kind == "strafe":
            pos = [-6 + 0.25 * f * abs(speed) / 0.025, height, radius * 0.8]
            tgt = target0 + np.array([pos[0] * 0.5, 0, 0])
        else:
            pos = [radius * math.cos(a0) * (1 - 0.012 * f), height, radius * math.sin(a0) * (1 - 0.012 * f)]
            tgt = target0
        cams.append(Camera.look_at(pos, tgt))
    return cams


def _render_pixels(scene, cam, H, W, ox, oy):
    d = cam.rays(H, W, ox, oy).reshape(-1, 3)
    o = cam.pos.expand_as(d)
    return scene.shade(o, d)


def _motion(cam, prev, p, H, W, ox=0.5, oy=0.5):
    """Motion vector (to previous frame, in pixels) and expected depth in previous camera."""
    j = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W).reshape(-1) + ox
    i = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W).reshape(-1) + oy
    u, v, z = prev.project(p, H, W)
    mv = torch.stack([u - j, v - i], -1)
    return mv, z


def render_sequence(seed: int, n_frames: int, lr_hw=(108, 192), ss: int = 3):
    """Render one sequence. Returns dict of float16 numpy arrays, channel-first per frame."""
    rng = np.random.default_rng(seed)
    scene = Scene(rng)
    cams = camera_path(rng, n_frames)
    Hl, Wl = lr_hw
    Hh, Wh = 2 * Hl, 2 * Wl
    out = {k: [] for k in ["lr_color", "lr_normal", "lr_albedo", "lr_depth", "lr_mv", "lr_pz",
                           "hr_color", "hr_depth", "hr_mv", "hr_pz", "jitter"]}
    for f, cam in enumerate(cams):
        prev = cams[f - 1] if f > 0 else cam
        jx, jy = halton(f + 1, 2), halton(f + 1, 3)
        rad, n, alb, p, depth, _ = _render_pixels(scene, cam, Hl, Wl, jx, jy)
        mv, pz = _motion(cam, prev, p, Hl, Wl, jx, jy)
        depth = ((p - cam.pos) * cam.fwd).sum(-1)  # view-space z, comparable with pz
        out["lr_color"].append(rad.T.reshape(3, Hl, Wl))
        out["lr_normal"].append(n.T.reshape(3, Hl, Wl))
        out["lr_albedo"].append(alb.T.reshape(3, Hl, Wl))
        out["lr_depth"].append(depth.reshape(1, Hl, Wl))
        out["lr_mv"].append(mv.T.reshape(2, Hl, Wl))
        out["lr_pz"].append(pz.reshape(1, Hl, Wl))
        out["jitter"].append(torch.tensor([jx, jy]))
        # HR ground truth: ss x ss stratified supersampling
        acc = torch.zeros(Hh * Wh, 3)
        for a in range(ss):
            for b in range(ss):
                r, *_ = _render_pixels(scene, cam, Hh, Wh, (b + 0.5) / ss, (a + 0.5) / ss)
                acc += r
        out["hr_color"].append((acc / ss ** 2).T.reshape(3, Hh, Wh))
        _, _, _, p_c, depth_c, _ = _render_pixels(scene, cam, Hh, Wh, 0.5, 0.5)
        mv_h, pz_h = _motion(cam, prev, p_c, Hh, Wh)
        depth_c = ((p_c - cam.pos) * cam.fwd).sum(-1)
        out["hr_depth"].append(depth_c.reshape(1, Hh, Wh))
        out["hr_mv"].append(mv_h.T.reshape(2, Hh, Wh))
        out["hr_pz"].append(pz_h.reshape(1, Hh, Wh))
    res = {k: torch.stack(v).numpy().astype(np.float16) for k, v in out.items()}
    # auto-exposure: key value 0.18 on log-average luminance of the HR frames
    hr = torch.stack(out["hr_color"])
    lum = 0.2126 * hr[:, 0] + 0.7152 * hr[:, 1] + 0.0722 * hr[:, 2]
    logavg = torch.exp(torch.log(lum.clamp(min=1e-4)).mean()).item()
    res["exposure"] = np.array(0.18 / logavg, dtype=np.float32)
    res["preset"] = np.array(scene.preset)
    return res
