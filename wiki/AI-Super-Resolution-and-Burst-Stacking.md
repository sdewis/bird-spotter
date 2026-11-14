# AI Super-Resolution & Burst Stacking

This page covers the technical implementation of Bird Spotter's computational photography and AI enhancement pipeline, designed to eliminate digital zoom pixelation and produce clean bird portraits.

---

## 1. The Digital Zoom Pixelation Challenge

Webcams and compact PTZ cameras like the OBSBOT Tiny SE feature fixed focal-length lenses. They have **no optical zoom glass**:

```text
At 1.0x Native Zoom: Full 1920x1080 sensor readout (2,073,600 physical pixels)
At 2.0x Digital Zoom: 960x540 sensor crop upscaled 2x (518,400 physical pixels)
At 3.0x Digital Zoom: 640x360 sensor crop upscaled 3x (230,400 physical pixels)
At 4.0x Digital Zoom: 480x270 sensor crop upscaled 4x (129,600 physical pixels)
```

At $4.0\times$ zoom, over **$93\%$ of the physical sensor pixels are discarded**, and the camera ISP bilinearly interpolates the remaining $480 \times 270$ pixels up to $1080\text{p}$. When combined with high camera hardware sharpness ($74/100$), every digital edge develops heavy white ringing halos and jagged pixel staircasing.

### The Laplacian Sharpness Trap

In earlier versions, burst frames were chosen by raw Laplacian variance:
$$\text{Var}(\nabla^2 I)$$
Because the Laplacian operator computes second-order spatial derivatives, artificial digital interpolation noise and high-contrast ringing edges produce massive mathematical variance spikes ($>5,000$), while clean, in-focus birds at lower zoom score $\sim 250 - 400$.

As a result, **the burst selector was actively biased towards choosing the most corrupted, pixelated frames**.

---

## 2. Solution 1: Noise-Robust Sharpness Metric

To prevent sensor grain and digital artifacts from inflating the sharpness score, `compute_crop_sharpness` applies a subtle Gaussian filter ($\sigma=0.6$) before the Laplacian operator:

```python
def compute_crop_sharpness(bgr_crop: np.ndarray) -> float:
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(gray, (3, 3), 0.6)
    return float(cv2.Laplacian(filtered, cv2.CV_64F).var())
```

### Empirical Metric Comparison

| Test Frame | Raw Laplacian | Filtered Laplacian | Tenengrad Gradient |
| :--- | :---: | :---: | :---: |
| **In-Focus Jackdaw ($1.5\times$ zoom)** | 311.6 | **124.1** | **8,843.7** |
| **Digital-Zoom Noise ($3.0\times$ zoom)** | 3,994.6 *(noise spike)* | **239.3** | **6,954.1** |

With the noise-robust pre-filter, the true optical focus of the subject determines the burst selection rather than high-frequency sensor noise.

---

## 3. Solution 2: Sub-Pixel Burst Stacking (Fusion)

During `SETTLING_BURST`, the camera captures a 5-frame burst. Because the camera is mounted on a mechanical gimbal and the bird makes subtle breathing movements, natural sub-pixel optical diversity exists across the frames.

```mermaid
flowchart LR
    F1["Burst Frame 1 (Ref)"]
    F2["Burst Frame 2"]
    F3["Burst Frame 3"]
    F4["Burst Frame 4"]
    F5["Burst Frame 5"]

    F1 & F2 & F3 & F4 & F5 --> Align["Sub-Pixel Phase Correlation\n(cv2.phaseCorrelate)"]
    Align --> Filter{"Response >= 0.45\n& Shift < 18%?"}
    Filter -- Yes --> Warp["Bilinear / Lanczos Warp"]
    Filter -- No --> Drop["Discard Outlier / Motion"]
    Warp --> Blend["Weighted Temporal Fusion\nSNR Gain ~ 2.2x (sqrt(5))"]
    Blend --> Fused["Fused Noise-Free Crop"]
```

### Alignment Mathematics

The translation offset $(\Delta x, \Delta y)$ between reference crop $I_1$ and candidate $I_2$ is computed via the cross-power spectrum:
$$R = \frac{\mathcal{F}\{I_1\} \cdot \mathcal{F}^*\{I_2\}}{\left| \mathcal{F}\{I_1\} \cdot \mathcal{F}^*\{I_2\} \right|}$$
$$r = \mathcal{F}^{-1}\{R\}$$

The peak of $r$ indicates the sub-pixel translation vector $(\Delta x, \Delta y)$, with normalized correlation peak height $C \in [0, 1]$.
* If $C \ge 0.45$ and displacement is small, the frame is warped and added to the fusion accumulator.
* If the bird suddenly turned its head or hopped, $C < 0.45$, and the outlier frame is safely dropped.

**Result**: Random sensor photon noise and JPEG macroblocking are attenuated by up to $\sqrt{N} \approx 2.23\times$, providing a pristine baseline for subsequent AI upscaling.

---

## 4. Solution 3: AI Super-Resolution (`SuperResolutionEnhancer`)

Bird Spotter incorporates deep-learning super-resolution natively using OpenCV's `cv2.dnn_superres` module.

### Neural Architectures Supported

1. **FSRCNN (Fast Super-Resolution CNN)**:
   * Uses shrinking, mapping, and expanding convolutional layers followed by a deconvolution layer.
   * Model file: `models/fsrcnn_x4.pb` ($\sim 41\text{ KB}$).
   * Upscales $4\times$ sub-pixel detail directly without bicubic pre-interpolation.
2. **ESPCN (Efficient Sub-Pixel CNN)**:
   * Extracts feature maps in the low-resolution domain and applies sub-pixel convolution (PixelShuffle) at the output.
   * Model file: `models/espcn_x4.pb` ($\sim 100\text{ KB}$).
   * Extremely lightweight.

### Live Performance Benchmarks (Raspberry Pi 5 CPU)

Tested on a $80 \times 80$ bird crop upscaled $4\times$ to $320 \times 320$:

| Model & Scale | Input Size | Output Size | Pi 5 Latency | Quality (PSNR) |
| :--- | :---: | :---: | :---: | :---: |
| **ESPCN $4\times$** | $80 \times 80$ | $320 \times 320$ | **$8.6\text{ ms}$** | 31.5 dB |
| **FSRCNN $4\times$** | $80 \times 80$ | $320 \times 320$ | **$19.4\text{ ms}$** | 31.9 dB |
| **Lanczos-4** | $80 \times 80$ | $320 \times 320$ | $<1.0\text{ ms}$ | 32.9 dB |
| **Old `INTER_AREA`** | $80 \times 80$ | $320 \times 320$ | $<1.0\text{ ms}$ | 29.2 dB *(Chunky/Pixelated)* |

*Note: AI super-resolution runs in the asynchronous photo-saving thread (`on_burst_capture_ready`), adding 0 ms delay to the real-time 30 FPS video feed.*

---

## 5. Noise-Gated Unsharp Masking

After neural upsampling, a noise-gated unsharp mask enhances fine feather textures and specular eye highlights:

```python
def unsharp_mask(bgr_img, sigma=1.0, strength=0.35, threshold=2):
    blurred = cv2.GaussianBlur(bgr_img, (0, 0), sigma)
    diff = cv2.absdiff(bgr_img, blurred)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) >= threshold
    sharpened = cv2.addWeighted(bgr_img, 1.0 + strength, blurred, -strength, 0)
    out = bgr_img.copy()
    out[mask] = sharpened[mask]
    return out
```

The `threshold=2` gate prevents sharpening flat background noise (like smooth sky or defocus blur), applying enhancement exclusively to edges and feather patterns.
