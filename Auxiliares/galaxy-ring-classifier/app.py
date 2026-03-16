"""
Galaxy Ring Classifier — Streamlit App
Classifies galactic rings (inner / outer / inner+outer) using a
fine-tuned Zoobot ConvNeXt-nano encoder.

Two modes:
  1. Batch CSV: upload a CSV with galaxy coordinates → bulk classification
  2. Single FITS: upload a .fits file → image + probabilities
"""

import os, io, re, tempfile, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import requests

from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.filters import gaussian

# ─── Config ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224
PIXSCALE   = 0.262
RING_TYPE  = ["inner", "outer", "inner+outer"]
RING_CODES = {4: "inner", 8: "outer", 12: "inner+outer"}
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Galaxy Ring Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Custom CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card-hover: #1a2235;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-orange: #f97316;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e293b;
}

.stApp {
    font-family: 'Outfit', sans-serif;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.app-header p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 300;
}

/* Result cards */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.prob-bar {
    height: 8px;
    border-radius: 4px;
    margin-top: 4px;
}

/* Metric boxes */
.metric-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-box .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-inner { background: rgba(59,130,246,0.2); color: #60a5fa; }
.badge-outer { background: rgba(249,115,22,0.2); color: #fb923c; }
.badge-both  { background: rgba(139,92,246,0.2); color: #a78bfa; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Model & helper classes (from notebook)
# =====================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        smooth = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth = smooth * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets].unsqueeze(1)
            focal_weight = focal_weight * alpha_t
        loss = -(focal_weight * smooth * log_probs).sum(dim=1)
        return loss.mean()


class ZoobotRingSubclassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.4,
                 encoder_name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano'):
        super().__init__()
        self.device = DEVICE
        self.ring_type = RING_TYPE
        self.n_classes = n_classes
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            encoder_dim = self.encoder(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.BatchNorm1d(encoder_dim), nn.Dropout(p=dropout),
            nn.Linear(encoder_dim, 128), nn.GELU(),
            nn.BatchNorm1d(128), nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, n_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        self.criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
        self.to(self.device)

    def forward(self, x):
        features = self.encoder(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.head(features)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def predict_probs(self, tensor):
        """Return class probabilities for a batch of tensors."""
        self.eval()
        with torch.no_grad():
            out = self(tensor.to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


# =====================================================================
# Transformations (from notebook)
# =====================================================================
from torchvision import transforms

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class Transformations:
    @staticmethod
    def log_n_scale_transform(img_data, log_a=1000):
        if img_data is None or img_data.size == 0:
            return None
        img = img_data.astype(np.float64)
        vmin, vmax = np.nanpercentile(img, [0.5, 99.5])
        img = np.clip(img, vmin, vmax)
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        img = np.log10(1 + log_a * img) / np.log10(1 + log_a)
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def rgi_lognorm_transform(r, g, z, log_a=1000):
        channels = []
        for band in [r, g, z]:
            ch = Transformations.log_n_scale_transform(band, log_a=log_a)
            if ch is None:
                return None
            channels.append(ch)
        return np.stack(channels, axis=-1)

    @staticmethod
    def apply_transform(r_band, g_band, z_band, method="rgi_lognorm"):
        if method == "rgi_lognorm":
            return Transformations.rgi_lognorm_transform(r_band, g_band, z_band)
        return Transformations.rgi_lognorm_transform(r_band, g_band, z_band)


# =====================================================================
# Morphological analysis functions (from notebook v5)
# =====================================================================

def compute_radial_profile(img, cy, cx, n_bins=80):
    h, w = img.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    r_map = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    max_r = min(cy, cx, h - cy, w - cx)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    profile = []
    for i in range(len(bin_edges) - 1):
        mask = (r_map >= bin_edges[i]) & (r_map < bin_edges[i + 1])
        profile.append(float(np.median(img[mask])) if mask.sum() > 0 else 0.0)
    radii_px = (bin_edges[:-1] + np.diff(bin_edges) / 2)
    radii_arc = radii_px * PIXSCALE
    return radii_px, radii_arc, np.array(profile)


def azimuthal_profile(img, center, radius_px, width=6, n_sectors=36):
    cy, cx = center
    h, w = img.shape
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    values = []
    for a in angles:
        sector_vals = []
        for dr in range(-width // 2, width // 2 + 1):
            r = radius_px + dr
            if r <= 0:
                continue
            xp = int(np.round(cx + r * np.cos(a)))
            yp = int(np.round(cy + r * np.sin(a)))
            if 0 <= xp < w and 0 <= yp < h:
                sector_vals.append(float(img[yp, xp]))
        if sector_vals:
            values.append(np.mean(sector_vals))
    if not values:
        return {"mean": 0, "std": 0, "cv": 999, "is_ring_like": False}
    arr = np.array(values)
    mean = float(arr.mean())
    std = float(arr.std())
    cv = std / (mean + 1e-10)
    return {"mean": mean, "std": std, "cv": cv, "values": arr, "is_ring_like": cv < 0.45}


def fit_ellipse_to_isophote(img, center, radius_px, tolerance=12, min_pts=12):
    cy, cx = center
    h, w = img.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    r_map = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    ring_mask = (r_map >= radius_px - tolerance) & (r_map <= radius_px + tolerance)
    if ring_mask.sum() < min_pts:
        return None
    ys = y_idx[ring_mask].astype(np.float32)
    xs = x_idx[ring_mask].astype(np.float32)
    ws = img[ring_mask].astype(np.float32)
    ws = np.clip(ws, 0, None)
    ws_sum = ws.sum()
    if ws_sum < 1e-10:
        return None
    probs = ws / ws_sum
    n_sample = min(len(xs), 500)
    idx_sample = np.random.choice(len(xs), size=n_sample, replace=False, p=probs)
    pts = np.stack([xs[idx_sample], ys[idx_sample]], axis=1).astype(np.float32)
    if len(pts) < 5:
        return None
    try:
        (ex, ey), (ma, mi), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    semi_a = max(ma, mi) / 2.0
    semi_b = min(ma, mi) / 2.0
    ecc = float(np.sqrt(1 - (semi_b / semi_a)**2)) if semi_a > 0 else 0.0
    return {
        "semi_a": float(semi_a), "semi_b": float(semi_b),
        "angle": (float(angle) + 90.0) % 180.0,
        "eccentricity": ecc, "cx": float(ex), "cy": float(ey),
    }


def get_galaxy_shape(img, cy, cx):
    h, w = img.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    r_eff = max(h, w) * 0.25
    mask = ((x_idx - cx)**2 + (y_idx - cy)**2) < r_eff**2
    weights = img[mask].flatten()
    if weights.sum() < 1e-10:
        return 0.0, 0.75, r_eff
    ys = y_idx[mask].flatten()
    xs = x_idx[mask].flatten()
    X = np.stack([xs - cx, ys - cy], axis=1).astype(float)
    cov = np.cov(X.T, aweights=np.clip(weights, 0, None))
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    ratio = float(np.clip(np.sqrt(abs(vals[1]) / (abs(vals[0]) + 1e-10)), 0.3, 1.0))
    r_eff_est = float(np.sqrt(abs(vals[0])))
    return angle, ratio, max(r_eff_est, 8.0)


def _find_ring_radii_from_profile(profile, radii_px, max_ring_r):
    cut = int(np.searchsorted(radii_px, max_ring_r))
    prof = profile[:cut]
    rpx = radii_px[:cut]
    if len(prof) < 8:
        return []
    win = max(5, len(prof) // 6)
    if win % 2 == 0:
        win += 1
    try:
        prof_s = savgol_filter(prof, win, 3)
    except Exception:
        prof_s = prof.copy()
    skip = max(2, int(len(prof_s) * 0.04))
    search = prof_s[skip:]
    dyn = search.max() - search.min()
    peaks_a, _ = find_peaks(search, distance=max(3, len(search) // 10),
                            prominence=max(dyn * 0.010, 1e-5), width=1)
    candidates = set(int(rpx[p + skip]) for p in peaks_a)
    d2 = np.gradient(np.gradient(prof_s))
    for idx in np.where((d2[:-1] < 0) & (d2[1:] >= 0))[0]:
        r = int(rpx[idx])
        if r >= int(radii_px[skip]):
            candidates.add(r)
    return sorted(r for r in candidates if 5 < r <= max_ring_r)


def _fallback_radius(profile, radii_px, max_ring_r):
    peak = profile.max()
    if peak < 1e-10:
        return int(radii_px[len(radii_px)//3]), int(radii_px[int(len(radii_px)*0.7)])
    def _r_at(f):
        below = np.where(profile <= peak * f)[0]
        return int(radii_px[below[0]]) if len(below) > 0 else int(radii_px[len(radii_px)//2])
    r_in = int(np.clip(_r_at(0.50), 6, max_ring_r * 0.45))
    r_out = int(np.clip(_r_at(0.20), r_in * 1.5, max_ring_r))
    return r_in, r_out


def _fit_ellipse_synthetic(img, cy, cx, radius_px, galaxy_angle, galaxy_ratio):
    h, w = img.shape
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    thickness = max(3, int(radius_px * 0.18))
    for dr in range(-thickness, thickness + 1):
        r = radius_px + dr
        if r <= 0:
            continue
        b = max(2, int(r * galaxy_ratio))
        cv2.ellipse(ring_mask, (int(cx), int(cy)), (int(r), int(b)),
                    int(galaxy_angle), 0, 360, 1, -1)
    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        pts = np.vstack(contours).squeeze()
        if len(pts) >= 5:
            try:
                (ex, ey), (ma, mi), angle = cv2.fitEllipse(pts)
                return float(max(ma, mi))/2, float(min(ma, mi))/2, (float(angle)+90.0)%180.0
            except Exception:
                pass
    b = max(2, int(radius_px * galaxy_ratio))
    return float(radius_px), float(b), (float(galaxy_angle)+90.0) % 180.0


def detect_rings_v5(img_rgb, predicted_class):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    h, w = gray_f.shape
    cy0, cx0 = h // 2, w // 2
    search_r = min(h, w) // 4
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    cmask = ((x_idx - cx0)**2 + (y_idx - cy0)**2) < search_r**2
    region = gray_f * cmask
    tw = region.sum()
    if tw > 1e-6:
        cy = int(np.clip(np.sum(y_idx * region) / tw, h * 0.15, h * 0.85))
        cx = int(np.clip(np.sum(x_idx * region) / tw, w * 0.15, w * 0.85))
    else:
        cy, cx = cy0, cx0
    smoothed = gaussian(gray_f, sigma=2.0)
    galaxy_angle, galaxy_ratio, _ = get_galaxy_shape(gray_f, cy, cx)
    max_r = min(cy, cx, h - cy, w - cx)
    max_ring_r = max_r * 0.82
    radii_px, radii_arc, profile = compute_radial_profile(gray_f, cy, cx, n_bins=80)
    raw_candidates = _find_ring_radii_from_profile(profile, radii_px, max_ring_r)
    validated = []
    for rpx in raw_candidates:
        az = azimuthal_profile(smoothed, (cy, cx), rpx, width=6)
        if az["is_ring_like"]:
            validated.append({"radius_px": int(rpx), "az": az})
    ellipses = []
    for cand in validated:
        rpx = cand["radius_px"]
        tol = max(8, int(rpx * 0.18))
        ell = fit_ellipse_to_isophote(smoothed, (cy, cx), rpx, tolerance=tol)
        if ell is None:
            sa, sb, ang = _fit_ellipse_synthetic(smoothed, cy, cx, rpx, galaxy_angle, galaxy_ratio)
            ell = {"semi_a": sa, "semi_b": sb, "angle": ang,
                   "eccentricity": 1 - sb/sa if sa > 0 else 0,
                   "cx": float(cx), "cy": float(cy)}
        ellipses.append({
            "semi_a": ell["semi_a"], "semi_b": ell["semi_b"],
            "angle": ell["angle"], "radius_px": rpx,
            "detected": True, "az_cv": cand["az"]["cv"],
        })
    ellipses.sort(key=lambda e: e["radius_px"])
    ellipses = [e for e in ellipses if 4 < e["radius_px"] <= max_ring_r]
    merged = []
    for e in ellipses:
        if not merged:
            merged.append(e)
        else:
            prev = merged[-1]
            if abs(e["radius_px"] - prev["radius_px"]) < max_ring_r * 0.15:
                if e.get("az_cv", 1) < prev.get("az_cv", 1):
                    merged[-1] = e
            else:
                merged.append(e)
    ellipses = merged
    r_fb_in, r_fb_out = _fallback_radius(profile, radii_px, max_ring_r)

    def _make_ring(rpx, rtype):
        tol = max(8, int(rpx * 0.18))
        ell = fit_ellipse_to_isophote(smoothed, (cy, cx), rpx, tolerance=tol)
        if ell:
            return {"semi_a": ell["semi_a"], "semi_b": ell["semi_b"],
                    "angle": ell["angle"], "radius_px": int(rpx),
                    "type": rtype, "detected": False}
        sa, sb, ang = _fit_ellipse_synthetic(smoothed, cy, cx, rpx, galaxy_angle, galaxy_ratio)
        return {"semi_a": sa, "semi_b": sb, "angle": ang,
                "radius_px": int(rpx), "type": rtype, "detected": False}

    result = []
    if predicted_class == "inner":
        if ellipses:
            e = ellipses[0].copy(); e["type"] = "Inner"; result = [e]
        else:
            result = [_make_ring(r_fb_in, "Inner")]
    elif predicted_class == "outer":
        if ellipses:
            e = ellipses[-1].copy(); e["type"] = "Outer"; result = [e]
        else:
            result = [_make_ring(r_fb_out, "Outer")]
    else:
        if len(ellipses) >= 2:
            ie, oe = ellipses[0].copy(), ellipses[-1].copy()
            if oe["radius_px"] > ie["radius_px"] * 1.4:
                ie["type"] = "Inner"; oe["type"] = "Outer"; result = [ie, oe]
            else:
                ie["type"] = "Inner"
                r_out = int(np.clip(ie["radius_px"] * 2.2, r_fb_out, max_ring_r))
                result = [ie, _make_ring(r_out, "Outer")]
        elif len(ellipses) == 1:
            e = ellipses[0]
            half = max_ring_r / 2.0
            if e["radius_px"] < half:
                e["type"] = "Inner"
                r_out = int(np.clip(max(e["radius_px"] * 2.2, r_fb_out), r_fb_out, max_ring_r))
                result = [e, _make_ring(r_out, "Outer")]
            else:
                e["type"] = "Outer"
                r_in = int(np.clip(e["radius_px"] * 0.38, r_fb_in, e["radius_px"] * 0.6))
                result = [_make_ring(r_in, "Inner"), e]
        else:
            result = [_make_ring(r_fb_in, "Inner"), _make_ring(r_fb_out, "Outer")]
    return result, (cy, cx)


def detect_bar_improved(img, center, max_r, n_radii=18):
    cy, cx = center
    r_probe = np.linspace(max_r * 0.06, max_r * 0.50, n_radii)
    r_vals, eps_vals, pa_vals = [], [], []
    for r in r_probe:
        ell = fit_ellipse_to_isophote(img, center, r, tolerance=max(3, int(r*0.12)), min_pts=20)
        if ell is None:
            continue
        eps_vals.append(ell["eccentricity"])
        pa_vals.append(ell["angle"])
        r_vals.append(r)
    r_vals, eps_vals, pa_vals = np.array(r_vals), np.array(eps_vals), np.array(pa_vals)
    if len(r_vals) < 4:
        return None
    win = min(5, len(eps_vals) - 1) | 1
    eps_s = uniform_filter1d(eps_vals, size=win) if len(eps_vals) >= win else eps_vals
    pa_s = uniform_filter1d(pa_vals, size=win) if len(pa_vals) >= win else pa_vals
    peak_idx = int(np.argmax(eps_s))
    peak_eps = float(eps_s[peak_idx])
    bar_r = float(r_vals[peak_idx])
    if peak_eps < 0.30:
        return None
    bar_pa = float(pa_s[peak_idx])
    bar_len = bar_r * 2 * PIXSCALE
    ell_at_bar = fit_ellipse_to_isophote(img, center, bar_r, tolerance=max(3, int(bar_r*0.15)))
    aspect = float(ell_at_bar["semi_a"] / max(ell_at_bar["semi_b"], 1e-3)) if ell_at_bar else 2.0
    return {
        "angle": bar_pa, "bar_radius_px": bar_r,
        "length_arcsec": bar_len, "aspect_ratio": aspect,
        "ellipticity_peak": peak_eps,
    }


def measure_bar_endpoints(img, center, bar_info):
    cy, cx = center
    angle_rad = np.radians(bar_info["angle"])
    r = bar_info["bar_radius_px"]
    x1 = cx + r * np.cos(angle_rad)
    y1 = cy + r * np.sin(angle_rad)
    x2 = cx - r * np.cos(angle_rad)
    y2 = cy - r * np.sin(angle_rad)
    return (x1, y1), (x2, y2)


# =====================================================================
# Helper: FITS → RGB image
# =====================================================================
def fits_to_rgb(fits_data, method="rgi_lognorm"):
    """Convert 3-band FITS (g,r,z) to RGB numpy array."""
    if fits_data.ndim == 3:
        if fits_data.shape[0] == 3:
            g, r, z = fits_data[0], fits_data[1], fits_data[2]
        else:
            g, r, z = fits_data[:, :, 0], fits_data[:, :, 1], fits_data[:, :, 2]
    elif fits_data.ndim == 2:
        g = r = z = fits_data
    else:
        return None
    rgb = Transformations.apply_transform(r, g, z, method=method)
    return rgb


def load_fits_from_bytes(file_bytes):
    """Load FITS from uploaded bytes."""
    with io.BytesIO(file_bytes) as buf:
        with fits.open(buf) as hdul:
            data = hdul[0].data
            if data is None and len(hdul) > 1:
                data = hdul[1].data
    return data.astype(np.float32) if data is not None else None


def download_fits_cutout(ra, dec, pixscale=0.262, size=224, bands="grz", layer="ls-dr10"):
    """Download FITS cutout from Legacy Survey."""
    url = (f"https://www.legacysurvey.org/viewer/fits-cutout"
           f"?ra={ra}&dec={dec}&layer={layer}"
           f"&pixscale={pixscale}&size={size}&bands={bands}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with io.BytesIO(resp.content) as buf:
                with fits.open(buf) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        return data.astype(np.float32)
    except Exception:
        pass
    return None


# =====================================================================
# Model loading (cached)
# =====================================================================
@st.cache_resource
def load_model(checkpoint_path=None):
    """Load or initialize model."""
    model = ZoobotRingSubclassifier(n_classes=3, dropout=0.4)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_model(checkpoint_path)
    model.eval()
    return model


# =====================================================================
# Prediction helpers
# =====================================================================
def predict_single_image(model, img_rgb):
    """Predict ring class from RGB numpy array (H,W,3)."""
    pil_img = Image.fromarray(img_rgb.astype(np.uint8)).convert("RGB")
    tensor = val_transform(pil_img).unsqueeze(0)
    probs = model.predict_probs(tensor)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "class": RING_TYPE[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {RING_TYPE[i]: float(probs[i]) for i in range(len(RING_TYPE))},
    }


def create_visualization(img_rgb, prediction, show_morphology=True):
    """Create matplotlib figure with ring detection overlay."""
    pred_class = prediction["class"]
    probs = prediction["probs"]

    rings, (cy, cx) = detect_rings_v5(img_rgb, pred_class)

    gray_f = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smoothed = gaussian(gray_f, sigma=2.0)
    max_r = min(cy, cx, img_rgb.shape[0] - cy, img_rgb.shape[1] - cx)
    bar_info = detect_bar_improved(smoothed, (cy, cx), max_r)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0a0e1a')

    ring_colors = {"Inner": "#00BFFF", "Outer": "#FF8C00"}
    title_color = {"inner": "#00BFFF", "outer": "#FF8C00", "inner+outer": "#8b5cf6"}

    # Panel 0: Original
    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=12, color="white", fontweight="bold")

    # Panel 1: Detection
    axes[1].imshow(img_rgb)
    axes[1].plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
    for ring in rings:
        rtype = ring["type"]
        color = ring_colors.get(rtype, "#FFFF00")
        ls = "-" if ring.get("detected", True) else "--"
        lw = 2.5 if ring.get("detected", True) else 2.0
        ep = Ellipse(xy=(cx, cy), width=ring["semi_a"] * 2, height=ring["semi_b"] * 2,
                     angle=ring["angle"], fill=False, edgecolor=color, linewidth=lw, linestyle=ls)
        axes[1].add_patch(ep)
        ef = Ellipse(xy=(cx, cy), width=ring["semi_a"] * 2, height=ring["semi_b"] * 2,
                     angle=ring["angle"], fill=True, facecolor=color, alpha=0.07)
        axes[1].add_patch(ef)
        tag = "" if ring.get("detected", True) else " ~"
        lbl_y = cy - ring["semi_b"] - 10
        axes[1].annotate(f"{rtype} (r={ring['radius_px']}px){tag}",
                         xy=(cx, max(5, lbl_y)), fontsize=9, color="white",
                         fontweight="bold", ha="center", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85))
    if bar_info is not None:
        (x1, y1), (x2, y2) = measure_bar_endpoints(smoothed, (cy, cx), bar_info)
        axes[1].plot([x1, x2], [y1, y2], color="lime", linewidth=3, alpha=0.85)
        axes[1].annotate(f"Bar {bar_info['length_arcsec']:.1f}\" PA={bar_info['angle']:.0f}°",
                         xy=(cx, cy + max_r * 0.55), fontsize=8, color="lime",
                         fontweight="bold", ha="center",
                         bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.65))
    axes[1].set_title(f"Pred: {pred_class} ({prediction['confidence']:.0%})",
                      fontsize=13, fontweight="bold",
                      color=title_color.get(pred_class, "white"))
    axes[1].axis("off")

    # Panel 2: Probabilities
    colors_bar = ["#3b82f6", "#f97316", "#8b5cf6"]
    prob_vals = [probs[c] for c in RING_TYPE]
    bars = axes[2].barh(RING_TYPE, prob_vals, color=colors_bar, height=0.5)
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Probabilities", fontsize=12, color="white", fontweight="bold")
    axes[2].set_facecolor('#0a0e1a')
    axes[2].tick_params(colors='white')
    for spine in axes[2].spines.values():
        spine.set_color('#1e293b')
    for bar, p in zip(bars, prob_vals):
        axes[2].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{p:.1%}", va="center", fontweight="bold", fontsize=11, color="white")

    for ax in axes:
        ax.set_facecolor('#0a0e1a')

    plt.tight_layout()
    return fig


# =====================================================================
# UI
# =====================================================================
st.markdown("""
<div class="app-header">
    <h1>🌌 Galaxy Ring Classifier</h1>
    <p>Morphological ring classification using Zoobot ConvNeXt-nano encoder</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Model checkpoint
    checkpoint_file = st.file_uploader(
        "Upload model checkpoint (.pt)",
        type=["pt"],
        help="Upload your trained ZoobotRingSubclassifier checkpoint"
    )

    checkpoint_path = None
    if checkpoint_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp.write(checkpoint_file.read())
        tmp.close()
        checkpoint_path = tmp.name
        st.success("✅ Checkpoint loaded")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This app classifies galactic rings into three categories:
    - **Inner** — ring structure close to the nucleus
    - **Outer** — ring in the outer disk region
    - **Inner+Outer** — both ring types present

    Built on a fine-tuned [Zoobot](https://github.com/mwalmsley/zoobot)
    ConvNeXt-nano encoder with 3-phase progressive training.

    **Data sources:** SDSS DR10, MaNGA DR9, Legacy Survey
    """)

    st.markdown("---")
    st.markdown("### 🔧 Parameters")
    pixscale = st.number_input("Pixel scale (arcsec/px)", value=0.262, step=0.001, format="%.3f")
    cutout_size = st.number_input("Cutout size (px)", value=224, step=1)
    layer = st.selectbox("Legacy Survey layer", ["ls-dr10", "ls-dr9"], index=0)

# Load model
model = load_model(checkpoint_path)

# Main tabs
tab_csv, tab_fits = st.tabs(["📋 Batch CSV Classification", "🔭 Single FITS / Cutout"])


# ─────────────────────────────────────────────────────────────────────
# Tab 1: CSV Batch
# ─────────────────────────────────────────────────────────────────────
with tab_csv:
    st.markdown("### Upload a CSV with galaxy coordinates")
    st.markdown("""
    Your CSV should contain columns for **RA** and **Dec** (right ascension, declination).
    Supported column names: `ra`/`objra`, `dec`/`objdec`, plus optional `objID`/`name` and `z` (redshift).
    """)

    col_example1, col_example2 = st.columns(2)
    with col_example1:
        st.markdown("**Format A — SDSS style:**")
        st.code("objID,ra,dec,z\n1237648721210769659,134.447,-0.1999,0.0282", language="csv")
    with col_example2:
        st.markdown("**Format B — MaNGA style:**")
        st.code("name,anillos,objra,objdec,nsa_z\nmanga-10001-1902,4,134.194,56.787,0.0254", language="csv")

    csv_file = st.file_uploader("Upload CSV", type=["csv", "tsv"], key="csv_upload")

    if csv_file:
        sep = "\t" if csv_file.name.endswith(".tsv") else ","
        df = pd.read_csv(csv_file, sep=sep)
        st.markdown(f"**Loaded {len(df)} rows** — preview:")
        st.dataframe(df.head(10), use_container_width=True)

        # Auto-detect columns
        cols_lower = {c.lower(): c for c in df.columns}
        ra_col = cols_lower.get("ra") or cols_lower.get("objra")
        dec_col = cols_lower.get("dec") or cols_lower.get("objdec")
        id_col = cols_lower.get("objid") or cols_lower.get("name") or cols_lower.get("id")
        z_col = cols_lower.get("z") or cols_lower.get("nsa_z") or cols_lower.get("redshift")
        ring_col = cols_lower.get("anillos")

        if ra_col and dec_col:
            st.success(f"Detected columns → RA: `{ra_col}`, Dec: `{dec_col}`"
                       + (f", ID: `{id_col}`" if id_col else "")
                       + (f", z: `{z_col}`" if z_col else ""))

            max_batch = st.slider("Max galaxies to process", 1, min(len(df), 200), min(len(df), 20))

            if st.button("🚀 Run Classification", type="primary", use_container_width=True):
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, row in df.head(max_batch).iterrows():
                    ra_val = float(row[ra_col])
                    dec_val = float(row[dec_col])
                    gal_id = str(row[id_col]) if id_col else f"galaxy_{i}"

                    status.markdown(f"⏳ Processing **{gal_id}** ({i+1}/{max_batch}) — downloading cutout...")
                    progress.progress((i) / max_batch)

                    fits_data = download_fits_cutout(ra_val, dec_val,
                                                    pixscale=pixscale, size=cutout_size,
                                                    layer=layer)

                    if fits_data is not None:
                        rgb = fits_to_rgb(fits_data)
                        if rgb is not None:
                            pred = predict_single_image(model, rgb)
                            ring_code = None
                            if ring_col and ring_col in df.columns:
                                ring_code = row[ring_col]
                            results.append({
                                "ID": gal_id,
                                "RA": ra_val,
                                "Dec": dec_val,
                                "z": float(row[z_col]) if z_col else None,
                                "Predicted Class": pred["class"],
                                "Confidence": pred["confidence"],
                                "P(inner)": pred["probs"]["inner"],
                                "P(outer)": pred["probs"]["outer"],
                                "P(inner+outer)": pred["probs"]["inner+outer"],
                                "True Label": RING_CODES.get(int(ring_code), str(ring_code)) if ring_code and pd.notna(ring_code) else None,
                            })
                        else:
                            results.append({"ID": gal_id, "RA": ra_val, "Dec": dec_val,
                                            "Predicted Class": "ERROR", "Confidence": 0})
                    else:
                        results.append({"ID": gal_id, "RA": ra_val, "Dec": dec_val,
                                        "Predicted Class": "NO DATA", "Confidence": 0})

                progress.progress(1.0)
                status.markdown("✅ **Classification complete!**")

                df_results = pd.DataFrame(results)
                st.markdown("### Results")
                st.dataframe(df_results, use_container_width=True)

                # Summary metrics
                valid = df_results[~df_results["Predicted Class"].isin(["ERROR", "NO DATA"])]
                if len(valid) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Processed", len(valid))
                    col2.metric("Inner", len(valid[valid["Predicted Class"] == "inner"]))
                    col3.metric("Outer", len(valid[valid["Predicted Class"] == "outer"]))
                    col4.metric("Inner+Outer", len(valid[valid["Predicted Class"] == "inner+outer"]))

                    # Accuracy if true labels available
                    if "True Label" in df_results.columns and df_results["True Label"].notna().any():
                        matched = valid[valid["True Label"].notna()]
                        if len(matched) > 0:
                            acc = (matched["Predicted Class"] == matched["True Label"]).mean()
                            st.metric("Accuracy vs True Labels", f"{acc:.1%}")

                    # Distribution chart
                    st.markdown("### Distribution")
                    fig_dist, ax = plt.subplots(figsize=(8, 3), facecolor='#0a0e1a')
                    counts = valid["Predicted Class"].value_counts()
                    colors = {"inner": "#3b82f6", "outer": "#f97316", "inner+outer": "#8b5cf6"}
                    bars = ax.bar(counts.index, counts.values,
                                  color=[colors.get(c, "#666") for c in counts.index])
                    ax.set_facecolor('#0a0e1a')
                    ax.tick_params(colors='white')
                    ax.set_ylabel("Count", color="white")
                    for spine in ax.spines.values():
                        spine.set_color('#1e293b')
                    for bar, val in zip(bars, counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                                str(val), ha='center', color='white', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_dist)

                # Download CSV
                csv_out = df_results.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv_out,
                                   "galaxy_ring_classification_results.csv", "text/csv",
                                   use_container_width=True)
        else:
            st.error("Could not detect RA/Dec columns. Please ensure your CSV has columns "
                     "named `ra`/`objra` and `dec`/`objdec`.")


# ─────────────────────────────────────────────────────────────────────
# Tab 2: Single FITS / Cutout
# ─────────────────────────────────────────────────────────────────────
with tab_fits:
    st.markdown("### Analyze a single galaxy")

    input_mode = st.radio("Input method", ["Upload FITS file", "Legacy Survey URL / Coordinates"],
                          horizontal=True)

    img_rgb = None
    galaxy_label = ""

    if input_mode == "Upload FITS file":
        fits_file = st.file_uploader("Upload a FITS file (3-band g,r,z)",
                                     type=["fits", "fit", "fits.gz"],
                                     key="fits_upload")
        if fits_file:
            galaxy_label = fits_file.name
            with st.spinner("Reading FITS..."):
                fits_data = load_fits_from_bytes(fits_file.read())
                if fits_data is not None:
                    img_rgb = fits_to_rgb(fits_data)
                    if img_rgb is None:
                        st.error("Could not create RGB image from FITS data.")
                else:
                    st.error("Could not read FITS data.")

    else:  # Coordinates
        col_ra, col_dec = st.columns(2)
        with col_ra:
            ra_input = st.number_input("RA (degrees)", value=134.44717, format="%.6f")
        with col_dec:
            dec_input = st.number_input("Dec (degrees)", value=-0.1999727, format="%.6f")

        if st.button("🔭 Download & Classify", type="primary"):
            galaxy_label = f"RA={ra_input:.4f} Dec={dec_input:.4f}"
            with st.spinner(f"Downloading cutout from Legacy Survey ({layer})..."):
                fits_data = download_fits_cutout(ra_input, dec_input,
                                                pixscale=pixscale, size=cutout_size,
                                                layer=layer)
                if fits_data is not None:
                    img_rgb = fits_to_rgb(fits_data)
                    if img_rgb is None:
                        st.error("Could not create RGB from downloaded FITS.")
                else:
                    st.error("Failed to download cutout. Check coordinates and try again.")

    # Show results if we have an image
    if img_rgb is not None:
        st.markdown("---")
        st.markdown(f"#### 🌀 {galaxy_label}")

        with st.spinner("Running classification..."):
            prediction = predict_single_image(model, img_rgb)

        # Metrics row
        pred_class = prediction["class"]
        badge_cls = {"inner": "badge-inner", "outer": "badge-outer", "inner+outer": "badge-both"}

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-box">
                <div class="label">Predicted Class</div>
                <div class="value">{pred_class}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-box">
                <div class="label">Confidence</div>
                <div class="value">{prediction['confidence']:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-box">
                <div class="label">P(inner)</div>
                <div class="value">{prediction['probs']['inner']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-box">
                <div class="label">P(outer)</div>
                <div class="value">{prediction['probs']['outer']:.4f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Visualization
        with st.spinner("Running morphological analysis..."):
            fig = create_visualization(img_rgb, prediction)
        st.pyplot(fig, use_container_width=True)

        # Radial profile
        with st.expander("📊 Radial Profile", expanded=False):
            gray_f = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            h, w = gray_f.shape
            cy, cx = h // 2, w // 2
            radii_px, radii_arc, profile = compute_radial_profile(gray_f, cy, cx)
            fig_prof, ax_p = plt.subplots(figsize=(10, 3), facecolor='#0a0e1a')
            ax_p.plot(radii_px, profile, color="#3b82f6", lw=1.5)
            ax_p.set_xlabel("Radius (px)", color="white")
            ax_p.set_ylabel("Median intensity", color="white")
            ax_p.set_facecolor('#0a0e1a')
            ax_p.tick_params(colors='white')
            for spine in ax_p.spines.values():
                spine.set_color('#1e293b')
            ax_p.grid(True, alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig_prof)

        # Raw image view
        with st.expander("🖼️ RGB Image (zoomable)", expanded=False):
            st.image(img_rgb, caption=galaxy_label, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding:1rem 0;">
    Galaxy Ring Classifier • Zoobot ConvNeXt-nano • SDSS + MaNGA
</div>
""", unsafe_allow_html=True)
