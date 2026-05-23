# Method Notes — Rotation Estimation from Optical Flow

## Why Optical Flow for Fasciculation?

Fasciculation appears in ultrasound video as a brief, localized muscle twitch. At the pixel level, this manifests as a small rotational displacement of tissue texture around a local center. Dense optical flow captures the full per-pixel velocity field between frames, making it suitable for detecting this kind of subtle motion.

---

## Farnebäck Dense Optical Flow

OpenCV's `calcOpticalFlowFarneback` implements the Farnebäck algorithm, which models local image neighbourhoods as polynomial expansions and estimates displacement by minimizing the difference between expansions of consecutive frames.

Key parameters used:

| Parameter | ALS Script | Synthetic Script | Effect |
|-----------|-----------|-----------------|--------|
| `pyr_scale` | 0.5 | 0.5 | Scale between pyramid levels |
| `levels` | 3 | 3 | Number of pyramid levels |
| `winsize` | 15 | 15 | Averaging window size |
| `iterations` | 20 | 5 | Iterations per pyramid level |
| `poly_n` | 5 | 300 | Pixel neighbourhood size |
| `poly_sigma` | 1.0 | 1.0 | Gaussian smoothing sigma |

The ALS script uses more iterations (`20` vs `5`) for better accuracy on noisier real ultrasound data.

---

## Curl as a Rotation Proxy

Given flow field `(vx, vy)`, the 2D curl at pixel `(r, c)` is:

```
curl(r, c) = ∂vy/∂x - ∂vx/∂y
```

Approximated with forward finite differences:

```
curl(r, c) ≈ [vy(r, c+1) - vy(r, c)] - [vx(r+1, c) - vx(r, c)]
```

A positive curl indicates counter-clockwise local rotation; negative indicates clockwise. This is a scalar field over the entire frame.

---

## Per-Ring Averaging

For ring size `s`, a square window of side `s` is placed at the ROI center `(cen_r, cen_c)`. The curl values within this window are summed and divided by `s²`:

```
rotation_s = (1/s²) · Σ_{i,j ∈ window_s} curl(i, j)
```

This gives the mean curl (rotation rate) within that ring region.

**Note:** Square windows are used as an approximation of circular annular regions. The outer ring of size `s` includes all pixels from rings `< s`, so this measures cumulative rotation within radius `s`, not just the annular shell.

---

## Normalization Pipeline

For each ring size `s`:

1. **Moving average** (window=3) applied to the raw per-frame rotation values to reduce noise
2. **Perimeter normalization**: divide by `8 × s` (perimeter of a square of side `s`)
3. **Radius scaling**: multiply by `s`

Steps 2 and 3 together simplify to dividing by `8`:

```
final_value = (rotation_s_smoothed / (8s)) × s = rotation_s_smoothed / 8
```

The radius scaling is retained in the pipeline for interpretability — it preserves the physical meaning of rotation magnitude scaling with ring size.

---

## Output as BBVI Initialization

The final CSV contains per-ring, per-frame rotation values. These are loaded by the `Fasciculation-Detection-Using-Synthetic-Data` script as the initial mean `μ_θ` of the variational posterior before Bayesian refinement begins.

The optical flow estimate provides a fast, non-iterative initialisation that places the BBVI optimization close to the correct solution, significantly reducing the number of EM iterations needed for convergence.
