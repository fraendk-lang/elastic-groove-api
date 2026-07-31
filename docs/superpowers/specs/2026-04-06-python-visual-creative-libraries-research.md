# Python Visual/Creative Libraries -- Deep Dive Research

> Context: FastAPI backends on Railway (headless Linux, no GPU unless noted)
> Date: 2026-04-06

---

## 1. rembg -- Background Removal

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install "rembg[cpu]"` wheel | ~44 KB (but pulls ~500 MB of deps via onnxruntime + numpy) |
| u2net model (auto-downloaded on first run) | ~176 MB |
| u2netp (lightweight) | ~4 MB |
| silueta | ~43 MB |
| isnet-general-use | ~176 MB |
| birefnet-general | ~200+ MB |
| Models stored at | `~/.u2net/` |

**RAM during inference:** ~1-2 GB for a typical 2000x2000 image with u2net on CPU. Alpha matting pushes this to ~4-10 GB. Keep max image dimension at 2048px for production safety.

**Processing time (CPU, 2000x2000):** Expect 10-20 seconds on a 2-core Railway instance. A 2784x1856 image benchmarked at ~14s on CPU. First run is slower due to model download + ONNX session initialization.

### Available Models (15+)

| Model | Best For | Quality |
|---|---|---|
| `u2net` | General purpose | Good, the default |
| `u2netp` | Speed over quality | Acceptable |
| `silueta` | Low-resource servers | Same arch as u2net, 43 MB |
| `isnet-general-use` | Diverse objects | Better than u2net on non-human subjects (IoU 0.82) |
| `isnet-anime` | Anime characters | High accuracy on illustrated characters |
| `u2net_human_seg` | People only | Optimized for human subjects |
| `birefnet-general` | Best overall quality | State-of-the-art edges (hair, glass, fabric) |
| `birefnet-portrait` | Portraits | Fine-grained portrait edges |
| `sam` | Interactive/any object | Facebook's Segment Anything |

**Recommendation for Elastic Paint/Prism:** Start with `isnet-general-use` for general use, offer `birefnet-general` as premium quality option.

### Practical Code Example

```python
from rembg import remove, new_session
from PIL import Image
import io

# Reuse session across requests (critical for performance)
session = new_session("isnet-general-use")

def remove_background(image_bytes: bytes) -> bytes:
    input_image = Image.open(io.BytesIO(image_bytes))
    # Cap dimensions to prevent OOM
    input_image.thumbnail((2048, 2048), Image.LANCZOS)
    output = remove(input_image, session=session)
    buf = io.BytesIO()
    output.save(buf, format="PNG")
    return buf.getvalue()
```

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from rembg import remove, new_session

app = FastAPI()
session = new_session("isnet-general-use")  # Load once at startup

@app.post("/api/remove-bg")
async def remove_bg(file: UploadFile):
    image_bytes = await file.read()
    result = remove(
        Image.open(io.BytesIO(image_bytes)),
        session=session,
    )
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
```

### Gotchas

1. **First-run model download:** The model downloads on first call. In Railway, pre-download in the Dockerfile or startup script, or the first user request will timeout.
2. **Memory spikes:** Large images + alpha matting = OOM. Always `thumbnail()` inputs.
3. **Session reuse is mandatory.** Creating a new session per request re-loads the ONNX model every time (~2-5s wasted).
4. **Python version:** Requires 3.11-3.13. If Railway's default Python is 3.10, you'll need to specify.
5. **onnxruntime CPU is large:** The full dependency tree is ~500 MB installed. Budget Railway disk accordingly.
6. **No significant GPU speedup:** CPU vs GPU difference is surprisingly small (14s vs 11s in benchmarks). CPU is fine for this use case.

### Comparison

| Tool | Pros | Cons |
|---|---|---|
| rembg (local) | Free, no API limits, offline | Heavy deps, ~15s per image |
| remove.bg API | Faster, better quality | $0.20/image, vendor lock-in |
| Replicate (birefnet) | Best quality, scales | ~$0.01-0.05/image, cold starts |
| Cloudflare Workers AI | Edge deployment | Limited model selection |

---

## 2. opencv-python-headless -- Image Processing

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install opencv-python-headless` | ~40 MB wheel (varies by platform) |
| RAM at import | ~30-50 MB |
| No system deps needed | Headless variant bundles everything |

The `-headless` variant is critical for Railway -- it excludes Qt/GTK GUI dependencies that would fail on a headless server.

### Most Useful Operations (Beyond Pillow)

**What Pillow can't do well that OpenCV excels at:**

```python
import cv2
import numpy as np

# 1. Canny Edge Detection -- core for creative "sketch" effects
edges = cv2.Canny(img, threshold1=50, threshold2=150)

# 2. Bilateral Filter -- smooth skin while preserving edges (beauty mode)
smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# 3. HSV color manipulation -- isolate and shift hues precisely
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180  # Shift hue by 30 degrees
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 4. Lab color space -- perceptual lightness adjustments
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l = clahe.apply(l)  # Adaptive histogram equalization on lightness only
result = cv2.merge([l, a, b])
result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# 5. Pencil sketch effect
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv = cv2.bitwise_not(gray)
blur = cv2.GaussianBlur(inv, (21, 21), 0)
sketch = cv2.divide(gray, 255 - blur, scale=256)

# 6. Stylization (built-in "painting" effect)
stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.07)

# 7. Edge-preserving filter (watercolor-like)
watercolor = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
```

### FastAPI Integration Pattern

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()

@app.post("/api/effects/sketch")
async def sketch_effect(file: UploadFile):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    _, buf = cv2.imencode('.png', sketch)
    return Response(content=buf.tobytes(), media_type="image/png")
```

### Gotchas

1. **BGR, not RGB.** OpenCV uses BGR channel order everywhere. Converting to/from Pillow (RGB) requires `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
2. **No built-in "style transfer" neural network.** `cv2.stylization()` is a non-photorealistic filter, not neural style transfer. For real style transfer, use a separate model.
3. **Memory: images are always NumPy arrays.** A 4000x3000 RGB image = ~36 MB in RAM. Multiple copies during processing stack up.
4. **Thread safety:** OpenCV functions are generally thread-safe, but some (like VideoCapture) are not.
5. **Import time:** `import cv2` takes ~0.3-0.5s. Acceptable.

### OpenCV vs scikit-image

| Criterion | OpenCV | scikit-image |
|---|---|---|
| Speed | Fast (C++ core) | Slower (pure Python + NumPy) |
| Install size | ~40 MB | ~25 MB |
| Creative effects | `stylization`, `edgePreservingFilter`, `pencilSketch` built-in | Better segmentation, watershed, morphology |
| Color spaces | BGR-centric, 150+ conversions | RGB-native, cleaner API |
| Use when | You need speed or built-in artistic filters | You need scientific image analysis or cleaner code |
| For Elastic tools | Primary choice | Use for specific algorithms (e.g., SLIC superpixels) |

**Verdict:** Use OpenCV as the workhorse. Pull in scikit-image only for specific algorithms like watershed segmentation or SLIC superpixels.

---

## 3. moderngl -- Headless GLSL Shaders

### The Hard Truth About Railway

**This is the most problematic library for your stack.** Railway containers are minimal Linux with no GPU. ModernGL requires OpenGL 3.3, which means you need:

1. Mesa3D with LLVMpipe software renderer
2. EGL libraries for headless context creation
3. System packages: `libgl1-mesa-dev`, `libegl1-mesa-dev`, `libgles2-mesa-dev`, `mesa-utils`

**Railway does NOT have Mesa pre-installed.** You'd need a custom Dockerfile:

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PYOPENGL_PLATFORM=egl

RUN pip install moderngl Pillow numpy
```

### Headless Context Setup

```python
import moderngl
import numpy as np
from PIL import Image

# Create standalone context with EGL backend (no display needed)
ctx = moderngl.create_context(standalone=True, backend='egl')

# Fragment shader: apply a vignette + color shift effect
prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_position;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_position * 0.5 + 0.5;
        }
    ''',
    fragment_shader='''
        #version 330
        uniform sampler2D tex;
        uniform float time;
        in vec2 uv;
        out vec4 fragColor;
        void main() {
            vec4 color = texture(tex, uv);
            float vignette = 1.0 - length(uv - 0.5) * 1.2;
            color.rgb *= vignette;
            color.r += sin(time) * 0.1;
            fragColor = color;
        }
    ''',
)

# Full-screen quad
vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(prog, vbo, 'in_position')

# Load image as texture
img = Image.open("input.jpg").convert("RGB")
texture = ctx.texture(img.size, 3, img.tobytes())
texture.use(0)

# Render to framebuffer
fbo = ctx.framebuffer(color_attachments=[ctx.texture(img.size, 3)])
fbo.use()
prog['time'].value = 0.5
vao.render(moderngl.TRIANGLE_STRIP)

# Read back result
data = fbo.color_attachments[0].read()
result = Image.frombytes("RGB", img.size, data)
result.save("output.png")
```

### Performance on Software Rendering

| Resolution | LLVMpipe (no GPU) | With GPU |
|---|---|---|
| 1920x1080, simple shader | ~50-200ms | ~1-5ms |
| 1920x1080, complex shader | ~200-1000ms | ~5-20ms |
| 4K, simple shader | ~200-800ms | ~2-10ms |

Software rendering is 50-100x slower than GPU. For simple shaders (vignette, color grading, distortion), it's still usable at ~100-200ms. Complex multi-pass shaders will be slow.

### Gotchas

1. **EGL context creation can silently fail** on Railway if Mesa isn't installed correctly. Always wrap in try/except and have a fallback.
2. **LLVMpipe is single-threaded by default.** Set `LP_NUM_THREADS=4` env var to use multiple cores.
3. **No compute shaders on LLVMpipe** -- Mesa's software renderer usually only supports OpenGL 3.3, not 4.3.
4. **Memory:** Each texture and framebuffer allocates RAM. A 1920x1080 RGBA texture = ~8 MB. Multiple passes multiply this.
5. **The moderngl PyPI wheel (~50 KB) is tiny,** but system deps (Mesa) add ~100-200 MB to Docker image.

### Recommendation

**For Elastic Forge/Prism:** Consider whether the complexity is worth it. For most image effects (color grading, distortion, blur), OpenCV or NumPy will be faster on CPU than software-rendered GLSL. ModernGL shines only if you need *exact GLSL shader compatibility* with your Three.js frontend (i.e., preview server-side what Forge renders client-side).

**Alternative:** Run GLSL shaders via `wgpu-py` (WebGPU for Python) or just do the math in NumPy. For Railway without GPU, NumPy-based effects will often outperform software OpenGL.

---

## 4. colour-science -- Color Math

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install colour-science` | 9.1 MB wheel |
| RAM at import | ~50-80 MB (loads lookup tables) |
| Dependencies | NumPy only (core) |

This is a *massive* library -- 130+ color models, 20+ delta E methods, spectral data, appearance models. You'll use maybe 5% of it.

### Oklch/Oklab Usage

```python
import colour
import numpy as np

# sRGB -> Oklab
rgb = np.array([0.4, 0.6, 0.8])  # Linear sRGB, range 0-1
xyz = colour.sRGB_to_XYZ(rgb)
oklab = colour.XYZ_to_Oklab(xyz)
# oklab = [L, a, b] where L=lightness, a=green-red, b=blue-yellow

# Oklab -> Oklch (cylindrical form -- better for palette generation)
L, a, b = oklab
C = np.sqrt(a**2 + b**2)       # Chroma
h = np.degrees(np.arctan2(b, a)) % 360  # Hue in degrees
oklch = (L, C, h)

# Generate perceptually uniform palette: same L and C, evenly spaced hues
def generate_palette(base_L=0.7, base_C=0.15, n_colors=5):
    hues = np.linspace(0, 360, n_colors, endpoint=False)
    palette = []
    for h in hues:
        a = base_C * np.cos(np.radians(h))
        b = base_C * np.sin(np.radians(h))
        oklab = np.array([base_L, a, b])
        xyz = colour.Oklab_to_XYZ(oklab)
        rgb = colour.XYZ_to_sRGB(xyz)
        rgb_clipped = np.clip(rgb, 0, 1)
        palette.append((rgb_clipped * 255).astype(int))
    return palette
```

### Delta E for "Similar Color" Search

```python
import colour

# Compare two colors using CIE 2000 (most perceptually accurate)
lab1 = np.array([50.0, 25.0, -10.0])  # CIE Lab values
lab2 = np.array([52.0, 23.0, -8.0])
delta = colour.delta_E(lab1, lab2, method='CIE 2000')
# delta < 1.0 = imperceptible, < 3.0 = barely noticeable, < 6.0 = noticeable

# For Oklab, Delta E is just Euclidean distance (much simpler!)
def oklab_distance(oklab1, oklab2):
    return np.sqrt(np.sum((np.array(oklab1) - np.array(oklab2))**2))
```

### Color Harmony Generation

```python
def complementary(hue): return (hue + 180) % 360
def triadic(hue): return [(hue + i * 120) % 360 for i in range(3)]
def split_complementary(hue): return [(hue + 150) % 360, (hue + 210) % 360]
def analogous(hue, spread=30): return [(hue + i * spread) % 360 for i in range(-2, 3)]
def tetradic(hue): return [(hue + i * 90) % 360 for i in range(4)]

# Use with Oklch for perceptually balanced results
def harmony_palette(base_rgb, harmony_fn, L=0.7, C=0.12):
    xyz = colour.sRGB_to_XYZ(np.array(base_rgb) / 255)
    oklab = colour.XYZ_to_Oklab(xyz)
    base_hue = np.degrees(np.arctan2(oklab[2], oklab[1])) % 360
    hues = harmony_fn(base_hue)
    return [oklch_to_rgb(L, C, h) for h in hues]
```

### Integration with Pillow/NumPy

```python
from PIL import Image
import numpy as np
import colour

# Convert entire image to Oklab for manipulation
img = np.array(Image.open("photo.jpg")) / 255.0
xyz = colour.sRGB_to_XYZ(img)       # Operates on entire array at once
oklab = colour.XYZ_to_Oklab(xyz)     # Shape: (H, W, 3)

# Increase lightness uniformly
oklab[:, :, 0] *= 1.1
oklab[:, :, 0] = np.clip(oklab[:, :, 0], 0, 1)

# Convert back
xyz_out = colour.Oklab_to_XYZ(oklab)
rgb_out = colour.XYZ_to_sRGB(xyz_out)
result = Image.fromarray((np.clip(rgb_out, 0, 1) * 255).astype(np.uint8))
```

### Gotchas

1. **Import is slow:** `import colour` takes 1-3 seconds due to module loading. Import at app startup, not per-request.
2. **sRGB linearization matters.** `colour.sRGB_to_XYZ` handles gamma correctly. Do NOT just divide by 255 and call it linear RGB.
3. **Gamut clipping:** Oklab can produce out-of-gamut sRGB values. Always `np.clip(rgb, 0, 1)` before display.
4. **Overkill for simple tasks.** If you just need hex-to-rgb or HSL rotation, use `colorsys` (stdlib) or a lighter library.
5. **The library name is `colour-science` on PyPI but you `import colour`.** Confusing.

### Lighter Alternative: `python-oklch`

If you only need Oklch and nothing else, `python-oklch` (~10 KB) is far lighter. But colour-science gives you delta E, spectral data, and appearance models if you ever need them.

---

## 5. blend-modes -- Layer Compositing

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install blend-modes` | ~20 KB |
| Dependencies | NumPy only |
| RAM overhead | Negligible beyond image arrays |

### All Available Blend Modes (14)

| Mode | Photoshop Equivalent |
|---|---|
| `normal` | Normal |
| `soft_light` | Soft Light |
| `hard_light` | Hard Light |
| `lighten_only` | Lighten |
| `darken_only` | Darken |
| `dodge` | Color Dodge |
| `multiply` | Multiply |
| `overlay` | Overlay |
| `difference` | Difference |
| `subtract` | Subtract |
| `addition` | Linear Dodge (Add) |
| `grain_extract` | (GIMP-specific) |
| `grain_merge` | (GIMP-specific) |
| `divide` | Divide |

**Missing:** Screen, Color Burn, Vivid Light, Pin Light, Exclusion, Hue, Saturation, Color, Luminosity.

### Practical Code Example

```python
from blend_modes import multiply, soft_light, overlay
from PIL import Image
import numpy as np

def blend_layers(bg_bytes, fg_bytes, mode="multiply", opacity=0.7):
    bg = np.array(Image.open(io.BytesIO(bg_bytes)).convert("RGBA")).astype(float)
    fg = np.array(Image.open(io.BytesIO(fg_bytes)).convert("RGBA")).astype(float)

    # Images must be same size
    fg_resized = np.array(
        Image.fromarray(fg.astype(np.uint8)).resize(
            (bg.shape[1], bg.shape[0])
        )
    ).astype(float)

    blend_fn = {"multiply": multiply, "soft_light": soft_light, "overlay": overlay}
    result = blend_fn[mode](bg, fg_resized, opacity)

    return Image.fromarray(result.astype(np.uint8))
```

### Performance on Large Images

For a 4000x3000 RGBA image (~48 MB in memory per layer):
- Blend operation: ~50-150ms (NumPy vectorized)
- Memory: ~150 MB (background + foreground + result)
- Acceptable for server-side compositing

### Gotchas

1. **PROJECT IS UNMAINTAINED.** Last commit was years ago. It works, but don't expect bug fixes.
2. **Value range is 0-255 as floats, NOT 0-1.** This is unusual and will trip you up.
3. **Both images must be identical dimensions.** Resize beforehand.
4. **Must be RGBA (4 channels).** RGB-only images will error silently or produce garbage.
5. **Alpha handling is basic.** It respects alpha for compositing but doesn't do pre-multiplied alpha correctly in all modes.

### DIY with NumPy (Often Better)

```python
import numpy as np

def np_multiply(bg, fg, opacity=1.0):
    """Multiply blend mode -- 3 lines, no dependency."""
    bg_f = bg.astype(np.float32) / 255
    fg_f = fg.astype(np.float32) / 255
    result = bg_f * fg_f
    blended = bg_f * (1 - opacity) + result * opacity
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)

def np_screen(bg, fg, opacity=1.0):
    """Screen blend -- not available in blend-modes!"""
    bg_f = bg.astype(np.float32) / 255
    fg_f = fg.astype(np.float32) / 255
    result = 1 - (1 - bg_f) * (1 - fg_f)
    blended = bg_f * (1 - opacity) + result * opacity
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)

def np_overlay(bg, fg, opacity=1.0):
    bg_f = bg.astype(np.float32) / 255
    fg_f = fg.astype(np.float32) / 255
    mask = bg_f < 0.5
    result = np.where(mask, 2 * bg_f * fg_f, 1 - 2 * (1 - bg_f) * (1 - fg_f))
    blended = bg_f * (1 - opacity) + result * opacity
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
```

### Recommendation

**For Elastic Paint/Canvas:** Roll your own blend modes with NumPy. It's ~20 lines per mode, you get exactly the modes you need (including Screen, Color Burn, etc.), proper 0-1 normalization, and no unmaintained dependency. The `blend-modes` package is fine for prototyping but shouldn't be a production dependency.

**Better alternative:** `blendmodes` (different package, note spelling) or `pilgram` for Instagram-style filters that include blending internally.

---

## 6. opensimplex + numba -- Generative Art

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install opensimplex` | ~30 KB |
| `pip install numba` | ~100 MB (includes LLVM) |
| `pip install opensimplex-loops` | ~20 KB (recommended for animation) |
| Numba first-JIT overhead | ~2-5s per function, then cached |
| RAM for 1920x1080 float64 noise | ~16 MB per frame |

### Flow Field Generation

```python
import opensimplex
import numpy as np
from PIL import Image

opensimplex.seed(42)

def generate_flow_field(width=1920, height=1080, scale=0.005, time=0.0):
    """Generate a flow field using 3D noise (2D space + time)."""
    x = np.arange(width) * scale
    y = np.arange(height) * scale
    xx, yy = np.meshgrid(x, y)

    # Angle field from noise
    angles = opensimplex.noise3array(xx, yy, np.full_like(xx, time))
    angles = angles * np.pi * 2  # Map to 0-2pi

    # Convert to vector field
    dx = np.cos(angles)
    dy = np.sin(angles)
    return dx, dy, angles

# Visualize as color
def noise_to_image(width=1920, height=1080, scale=0.003):
    x = np.linspace(0, width * scale, width)
    y = np.linspace(0, height * scale, height)
    xx, yy = np.meshgrid(x, y)
    noise = opensimplex.noise2array(xx, yy)
    # Normalize to 0-255
    normalized = ((noise + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode='L')
```

### Animated Looping Noise (4D)

```python
from opensimplex_loops import looping_animated_2D_image
import numpy as np

# Generate 60 frames of seamlessly looping noise
noise_frames = looping_animated_2D_image(
    N_frames=60,
    N_pixels_x=1920,
    N_pixels_y=1080,
    t_step=0.1,     # Controls animation smoothness
    x_step=0.01,    # Controls spatial frequency
    y_step=0.01,
    dtype=np.float32  # Half the memory of float64
)
# noise_frames.shape = (60, 1080, 1920), values in [-1, 1]
```

### Performance (with and without Numba)

| Operation | Without Numba | With Numba (after JIT) |
|---|---|---|
| 1920x1080 2D noise | ~8-15s | ~40-80ms |
| 1920x1080 3D noise (flow field) | ~20-40s | ~100-200ms |
| 60 frames looping 1080p | ~minutes | ~5-15s |
| First call (JIT compilation) | N/A | +2-5s one-time cost |

**Numba is mandatory for production.** Without it, generating a single 1080p noise texture takes 10+ seconds. With Numba, it's real-time.

### FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import Response
import opensimplex
import numpy as np
from PIL import Image
import io

app = FastAPI()
opensimplex.seed(42)

@app.get("/api/noise")
async def generate_noise(
    width: int = 1920, height: int = 1080,
    scale: float = 0.005, seed: int = 42,
    octaves: int = 1
):
    opensimplex.seed(seed)
    x = np.linspace(0, width * scale, width)
    y = np.linspace(0, height * scale, height)
    xx, yy = np.meshgrid(x, y)

    noise = opensimplex.noise2array(xx, yy)

    # FBM (fractal Brownian motion) for richer textures
    if octaves > 1:
        for i in range(1, octaves):
            freq = 2 ** i
            amp = 0.5 ** i
            noise += amp * opensimplex.noise2array(xx * freq, yy * freq)

    normalized = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    img = Image.fromarray(normalized, mode='L')

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
```

### Gotchas

1. **Numba install is heavy** (~100 MB) and requires LLVM. On Railway, this adds to build time and image size.
2. **First-call JIT penalty.** The first request after deploy will be slow (2-5s extra). Warm up in startup hook.
3. **opensimplex-loops is a separate package** from opensimplex. They're complementary but install separately.
4. **Memory for animation:** 60 frames at 1920x1080 float32 = ~475 MB. Generate frame-by-frame if memory is tight.
5. **No GPU acceleration.** Numba uses CPU multi-threading. For GPU noise, you'd need moderngl or cupy.

---

## 7. vtracer -- Image Vectorization

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install vtracer` | ~1 MB wheel (Rust binary via PyO3) |
| Dependencies | None (self-contained) |
| RAM during processing | ~50-200 MB depending on image size and color_precision |

### Quality Comparison to Potrace

| Feature | vtracer | potrace |
|---|---|---|
| Color support | Full color (up to 256 colors) | Monochrome only |
| Algorithm complexity | O(n) | O(n^2) |
| Speed on large images | Much faster | Slower |
| Edge quality | Smooth splines, sometimes soft | Sharp, precise edges |
| Path optimization | Stacking strategy (compact SVGs) | Classic path tracing |
| Best for | Photos, illustrations, logos | Clean line art, high-contrast |
| Python package | `vtracer` | `pypotrace` (requires libpotrace) |

**Verdict:** vtracer wins for colored images and speed. Potrace wins for black-and-white precision work.

### Parameter Tuning Guide

```python
import vtracer

# High detail (photo-like)
svg_detailed = vtracer.convert_raw_image_to_svg(
    img_bytes,
    img_format="jpg",
    colormode="color",
    mode="spline",            # Smooth curves (use "polygon" for pixel art)
    filter_speckle=2,         # Keep small details
    color_precision=8,        # 2^8 = 256 colors
    layer_difference=8,       # More color layers
    corner_threshold=60,      # Angle for sharp corners
    length_threshold=4.0,     # Path segment length
    splice_threshold=45,      # Spline joining angle
    max_iterations=10,
    path_precision=5,         # Decimal places in SVG paths
)

# Poster style (simplified)
svg_poster = vtracer.convert_raw_image_to_svg(
    img_bytes,
    img_format="jpg",
    colormode="color",
    mode="spline",
    filter_speckle=10,        # Remove small patches
    color_precision=4,        # 2^4 = 16 colors
    layer_difference=32,      # Fewer layers
    corner_threshold=60,
    length_threshold=6.0,
)

# Clean line art (from edge-detected image)
svg_lineart = vtracer.convert_raw_image_to_svg(
    edge_bytes,
    img_format="png",
    colormode="binary",       # Black and white
    mode="spline",
    filter_speckle=4,
    corner_threshold=90,
)
```

### Processing Time

| Image | Poster (16 colors) | Detailed (256 colors) |
|---|---|---|
| 800x600 photo | ~0.5s | ~2-5s |
| 1920x1080 photo | ~1-3s | ~5-15s |
| 4000x3000 photo | ~5-10s | ~20-60s |
| Simple logo 500x500 | ~0.1s | ~0.5s |

### Output SVG Size

| Style | Typical SVG Size |
|---|---|
| Poster (16 colors) | 50-200 KB |
| Detailed (256 colors) | 500 KB - 5 MB |
| Binary line art | 10-100 KB |

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import vtracer

app = FastAPI()

@app.post("/api/vectorize")
async def vectorize(
    file: UploadFile,
    color_precision: int = 6,
    filter_speckle: int = 4,
    mode: str = "spline",
):
    img_bytes = await file.read()
    fmt = file.filename.split(".")[-1].lower()
    if fmt == "jpg": fmt = "jpeg"

    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes,
        img_format=fmt,
        colormode="color",
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
    )
    return Response(content=svg_str, media_type="image/svg+xml")
```

### Gotchas

1. **No streaming output.** The entire SVG is generated in memory before returning. Large/complex images can produce 5+ MB SVGs.
2. **No progress callback.** Long vectorizations appear to hang.
3. **`convert_raw_image_to_svg` expects raw file bytes, not a PIL Image.** Read the file bytes directly.
4. **`img_format` is picky.** Use "jpeg" not "jpg" for JPEG files.
5. **High color_precision + large images = slow.** Cap at color_precision=6 for responsive API.

---

## 8. svgwrite + cairosvg -- Vector Pipeline

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install svgwrite` | ~100 KB (pure Python) |
| `pip install cairosvg` | ~50 KB wheel |
| System dep: libcairo2 | ~2-5 MB (must be installed on Railway) |
| RAM for SVG operations | Minimal (~10 MB for complex SVGs) |

### Railway System Dependencies

**Critical: CairoSVG requires `libcairo2` system library.**

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    libffi-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN pip install svgwrite cairosvg
```

Or in Railway's `nixpacks.toml`:

```toml
[phases.setup]
aptPkgs = ["libcairo2-dev", "libpango1.0-dev", "libgdk-pixbuf2.0-dev", "fonts-dejavu-core"]
```

### Generate SVG Programmatically -> Render to PNG

```python
import svgwrite
import cairosvg
import io

def generate_generative_art(width=1920, height=1080, seed=42):
    """Generate a geometric pattern as SVG, render to PNG."""
    import random
    random.seed(seed)

    dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"))
    dwg.viewbox(0, 0, width, height)

    # Background
    dwg.add(dwg.rect((0, 0), (width, height), fill="#1a1a2e"))

    # Generative circles
    for _ in range(200):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = random.randint(5, 80)
        opacity = random.uniform(0.05, 0.4)
        hue = random.randint(180, 280)
        dwg.add(dwg.circle(
            center=(cx, cy), r=r,
            fill=f"hsl({hue}, 70%, 60%)",
            opacity=opacity,
        ))

    svg_string = dwg.tostring()

    # Render to PNG at desired resolution
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode(),
        output_width=width,
        output_height=height,
        dpi=150,
    )
    return png_bytes

# Scale-independent: render at 4K from the same SVG
png_4k = cairosvg.svg2png(
    bytestring=svg_string.encode(),
    output_width=3840,
    output_height=2160,
)
```

### Font Handling on Headless Railway

```python
# Fonts must be INSTALLED on the system. CairoSVG cannot use @font-face CSS.
# Install fonts in Dockerfile:
# RUN apt-get install -y fonts-dejavu-core fonts-liberation fonts-noto

# Use installed font names in SVG
dwg.add(dwg.text(
    "Hello World",
    insert=(100, 100),
    font_family="DejaVu Sans",  # Must match installed font name
    font_size="48px",
    fill="white",
))

# For custom fonts (.ttf), copy them to /usr/share/fonts/ and run fc-cache:
# COPY ./fonts/*.ttf /usr/share/fonts/custom/
# RUN fc-cache -f -v
```

### Gotchas

1. **`libcairo2` is the #1 deployment blocker.** Without it, `import cairosvg` fails with "no library called 'libcairo-2'". Always test in a Railway-like Docker container.
2. **Font availability is limited.** Only system-installed fonts work. No @font-face, no Google Fonts URLs. Pre-install any fonts you need.
3. **SVG feature support is incomplete.** CairoSVG doesn't support all SVG features (e.g., some filters, animations, foreignObject). Test your specific SVGs.
4. **Pango dependency for text layout.** Without `libpango1.0-dev`, text rendering may fail or look wrong.
5. **svgwrite has no validation.** It happily generates invalid SVG. Use a viewer to check output.

### Performance

- Generating a 200-element SVG: ~1-5ms
- Rendering to 1920x1080 PNG: ~50-200ms
- Rendering to 4K PNG: ~200-500ms
- Very complex SVGs (10K+ paths): 1-5s render time

### Recommendation for Elastic Tools

**Excellent for:** Generative art output, certificate/badge generation, data visualization export, resolution-independent graphics.

**Not suitable for:** Photo effects (use OpenCV), real-time rendering (use frontend Canvas/SVG), complex gradients with many stops.

---

## 9. Replicate API -- Remote AI

### Pricing Model

| Tier | Cost |
|---|---|
| CPU (Small) | $0.000025/sec ($0.09/hr) |
| Nvidia T4 | $0.000225/sec ($0.81/hr) |
| Nvidia A100 (80GB) | $0.001400/sec ($5.04/hr) |
| FLUX 1.1 Pro | $0.04/image (fixed) |
| Real-ESRGAN | ~$0.0055/run (~181 runs/$1) |
| SDXL | ~$0.01-0.05/image depending on steps |

**Billing:** Prepaid credits for new accounts (since July 2025). No free tier beyond initial credits.

### Latency for Common Models

| Model | Cold Start | Warm Inference |
|---|---|---|
| SDXL (1024x1024, 25 steps) | 15-60s | 5-15s |
| FLUX 1.1 Pro | 5-20s | 3-8s |
| Real-ESRGAN (2x upscale) | 5-15s | 0.5-2s |
| ControlNet | 15-45s | 8-20s |
| BiRefNet (bg removal) | 10-30s | 2-5s |

**Cold starts are the killer.** Models that haven't been called recently take 15-60s to boot. Replicate keeps popular models warm, but less-used ones sleep.

### Best Models for Creative Tools (2026)

| Use Case | Model | Why |
|---|---|---|
| Image generation | `black-forest-labs/flux-2-pro` | Best quality/speed ratio, consistent style |
| Quick generation | `black-forest-labs/flux-1.1-pro-ultra` | Ultra-fast, good quality |
| Upscaling | `batouresearch/magic-image-refiner` | Flexible: upscale, refine, or inpaint |
| Fast upscaling | `nightmareai/real-esrgan` | Fastest, reliable for bulk |
| Background removal | `lucataco/remove-bg` or BiRefNet | Better quality than local rembg |
| Style transfer | `fofr/face-to-many` | Artistic transformations |
| Inpainting | `ideogram-ai/ideogram-v3` | Natural typography + style edits |
| Img2Img | `stability-ai/sdxl` with img2img | Wide parameter control |

### FastAPI Integration (Async)

```python
from fastapi import FastAPI
import replicate
import asyncio
import httpx

app = FastAPI()
# Set REPLICATE_API_TOKEN env var

@app.post("/api/generate")
async def generate_image(prompt: str, width: int = 1024, height: int = 1024):
    """Async image generation with Replicate."""
    # Option 1: Async client (blocking but simple)
    output = await asyncio.to_thread(
        replicate.run,
        "black-forest-labs/flux-1.1-pro",
        input={"prompt": prompt, "width": width, "height": height},
    )
    # output is a URL to the generated image

    # Download the result
    async with httpx.AsyncClient() as client:
        resp = await client.get(output)
        return Response(content=resp.content, media_type="image/png")


@app.post("/api/upscale")
async def upscale_image(image_url: str, scale: int = 2):
    """Upscale with Real-ESRGAN."""
    output = await asyncio.to_thread(
        replicate.run,
        "nightmareai/real-esrgan",
        input={"image": image_url, "scale": scale},
    )
    return {"result_url": output}
```

### Webhook Pattern (For Long-Running Jobs)

```python
@app.post("/api/generate-async")
async def generate_async(prompt: str):
    """Start generation, get result via webhook."""
    prediction = await asyncio.to_thread(
        replicate.predictions.create,
        model="black-forest-labs/flux-2-pro",
        input={"prompt": prompt},
        webhook="https://your-railway-app.up.railway.app/api/webhook",
        webhook_events_filter=["completed"],
    )
    return {"prediction_id": prediction.id, "status": "processing"}

@app.post("/api/webhook")
async def replicate_webhook(request: Request):
    """Receive completed prediction from Replicate."""
    body = await request.json()
    prediction_id = body["id"]
    output_url = body["output"]
    # Store result, notify frontend via WebSocket, etc.
    return {"ok": True}
```

### Gotchas

1. **Cold starts kill UX.** Users see 30+ second waits for uncommon models. Mitigate: use popular models, or use Replicate's "keep warm" feature (costs money).
2. **Output URLs expire after 1 hour.** Download and store results immediately.
3. **Rate limits exist.** Default is ~50 concurrent predictions. Burst traffic will queue.
4. **No way to estimate cost programmatically** before running a prediction.
5. **Image files need to be URLs or base64.** You can't upload a file directly. Use a presigned S3/R2 URL or data URI.
6. **The Python client is sync by default.** Wrap in `asyncio.to_thread()` for FastAPI, or use `replicate.async_run()`.

### Recommendation for Elastic Ecosystem

**Use Replicate for:** Features you can't do locally (image generation, super-resolution, style transfer). Expose as "premium" features in Elastic Paint/Prism with a "generating..." loading state.

**Don't use Replicate for:** Anything you can do locally fast enough (bg removal with rembg, filters with OpenCV, vectorization with vtracer). The latency and cost add up.

---

## 10. albumentations -- Creative Effects

### Install Size & RAM Footprint

| Component | Size |
|---|---|
| `pip install albumentations` | ~2 MB |
| Dependencies | OpenCV, NumPy, scipy, scikit-image (pulls ~150 MB total) |
| RAM at import | ~50 MB (because of OpenCV) |

**Note:** As of 2025, development has moved to `AlbumentationsX` (next-gen fork). The original `albumentations` package still works but may not receive updates.

### Most Visually Interesting Effects for Creative Tools

```python
import albumentations as A
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg"))

# --- Weather Effects ---
rain = A.RandomRain(
    brightness_coefficient=0.9,
    drop_width=1, drop_length=20,
    blur_value=3, rain_type="heavy",
    p=1.0
)(image=img)["image"]

fog = A.RandomFog(
    fog_coef_lower=0.3, fog_coef_upper=0.6,
    alpha_coef=0.1, p=1.0
)(image=img)["image"]

snow = A.RandomSnow(
    snow_point_lower=0.1, snow_point_upper=0.3,
    brightness_coeff=2.5, p=1.0
)(image=img)["image"]

sun_flare = A.RandomSunFlare(
    src_radius=200,
    num_flare_circles_lower=3,
    num_flare_circles_upper=7,
    p=1.0
)(image=img)["image"]

# --- Lens Effects ---
chromatic = A.ChromaticAberration(
    primary_distortion_limit=0.05,
    secondary_distortion_limit=0.03,
    mode="green_purple",  # or "red_blue"
    p=1.0
)(image=img)["image"]

defocus = A.Defocus(
    radius=(5, 8),  # Blur radius range
    alias_blur=(0.1, 0.5),
    p=1.0
)(image=img)["image"]

# --- Artistic ---
emboss = A.Emboss(alpha=(0.5, 1.0), strength=(0.5, 1.0), p=1.0)(image=img)["image"]
posterize = A.Posterize(num_bits=3, p=1.0)(image=img)["image"]
solarize = A.Solarize(threshold=128, p=1.0)(image=img)["image"]
sepia = A.ToSepia(p=1.0)(image=img)["image"]
film_grain = A.FilmGrain(intensity_range=(0.3, 0.7), p=1.0)(image=img)["image"]

# --- Distortion ---
glass = A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=1.0)(image=img)["image"]
elastic = A.ElasticTransform(alpha=120, sigma=6, p=1.0)(image=img)["image"]
```

### Chaining Effects (Pipeline)

```python
# Create a "vintage film" preset
vintage = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.05), contrast_limit=0.2, p=1.0),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=-30, val_shift_limit=0, p=1.0),
    A.ToSepia(p=0.5),
    A.FilmGrain(intensity_range=(0.2, 0.5), p=1.0),
    A.Vignetting(p=1.0),
    A.ImageCompression(quality_range=(60, 80), p=0.7),
])

result = vintage(image=img)["image"]

# Create a "dream sequence" preset
dream = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.RandomGamma(gamma_limit=(120, 160), p=1.0),
    A.ChromaticAberration(primary_distortion_limit=0.02, p=1.0),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
])
```

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import Response
import albumentations as A
import numpy as np
from PIL import Image
import io

app = FastAPI()

PRESETS = {
    "vintage": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=-0.05, contrast_limit=0.15, p=1),
        A.HueSaturationValue(sat_shift_limit=-25, p=1),
        A.FilmGrain(intensity_range=(0.2, 0.4), p=1),
        A.Vignetting(p=1),
    ]),
    "rain": A.Compose([
        A.RandomRain(drop_width=1, drop_length=20, blur_value=3, p=1),
    ]),
    "chromatic": A.Compose([
        A.ChromaticAberration(primary_distortion_limit=0.04, mode="red_blue", p=1),
    ]),
}

@app.post("/api/effects/{preset_name}")
async def apply_effect(preset_name: str, file: UploadFile):
    if preset_name not in PRESETS:
        return {"error": f"Unknown preset. Available: {list(PRESETS.keys())}"}

    img = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))
    result = PRESETS[preset_name](image=img)["image"]

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
```

### Weather Effect Quality Assessment

| Effect | Quality | Notes |
|---|---|---|
| RandomRain | 6/10 | Looks decent, but rain drops are straight lines -- not photorealistic |
| RandomFog | 7/10 | `AtmosphericFog` (newer) is better than `RandomFog` |
| RandomSnow | 5/10 | Basically white dots; very basic |
| RandomSunFlare | 7/10 | Convincing lens flare circles |
| ChromaticAberration | 8/10 | Genuinely good, physically-based distortion |
| FilmGrain | 7/10 | Realistic grain pattern |

### Gotchas

1. **Designed for ML augmentation, not creative effects.** Parameters are randomized by default (ranges, not exact values). For deterministic effects, set both bounds to the same value or use `p=1.0`.
2. **`A.Compose` applies transforms in order** but each has its own probability (`p`). Set `p=1.0` for guaranteed application.
3. **Input must be NumPy uint8 RGB.** No RGBA support in most transforms. Strip alpha first, apply effect, re-attach alpha.
4. **Some transforms change image dimensions** (crops, rotations). Use `A.Resize()` at the end if you need fixed output size.
5. **Heavy dependency tree:** Pulls in OpenCV, scikit-image, scipy. If you already have OpenCV, the marginal cost is just scipy + scikit-image (~50 MB).
6. **Migration to AlbumentationsX:** The original package may stop receiving updates. Watch for breaking changes if you migrate later.

### Comparison to Manual OpenCV Effects

| Feature | Albumentations | DIY OpenCV |
|---|---|---|
| Weather effects | Built-in, one-liner | 20-50 lines each |
| Chromatic aberration | Excellent implementation | Hard to get right manually |
| Chaining/compose | First-class `Compose` pipeline | Manual orchestration |
| Deterministic control | Awkward (designed for randomness) | Full control |
| RGBA support | Poor | Full support |
| Dependency weight | Heavy | Just OpenCV |

**Recommendation:** Use albumentations for weather/atmospheric effects and chromatic aberration. Use OpenCV directly for color grading, blur, and sketch effects where you need precise control.

---

## Summary: Recommended Stack for Elastic Ecosystem

### Tier 1: Install These (Essential)

| Library | For | Install |
|---|---|---|
| `opencv-python-headless` | Image processing, filters, color spaces | `pip install opencv-python-headless` |
| `rembg[cpu]` | Background removal | `pip install "rembg[cpu]"` |
| `vtracer` | Image vectorization | `pip install vtracer` |
| `colour-science` | Color math, palettes, delta E | `pip install colour-science` |

### Tier 2: Install As Needed

| Library | For | Notes |
|---|---|---|
| `albumentations` | Weather effects, chromatic aberration, film grain | Heavy deps, but excellent effects |
| `svgwrite` + `cairosvg` | Generative art, SVG pipeline | Needs `libcairo2` system dep |
| `opensimplex` + `numba` | Noise textures, flow fields | Numba is heavy (~100 MB) |
| `replicate` | AI features (generation, super-res, style transfer) | Per-use cost, cold start latency |

### Tier 3: Skip or DIY

| Library | Recommendation |
|---|---|
| `blend-modes` | Roll your own with NumPy (10 lines per mode, more modes, maintained) |
| `moderngl` | Skip on Railway without GPU. Use OpenCV/NumPy for image effects. Only consider if you need GLSL shader compatibility with Elastic Forge frontend |

### Total Disk Budget (Tier 1 + 2)

| Component | Size |
|---|---|
| Python packages | ~700 MB |
| System deps (libcairo2, fonts) | ~50 MB |
| AI models (rembg) | ~200 MB |
| Numba LLVM | ~100 MB |
| **Total** | **~1 GB** |

Railway's free tier gives you 1 GB RAM and 1 GB disk. You'll need a paid plan ($5/mo) for this stack.

---

Sources:
- [rembg GitHub](https://github.com/danielgatis/rembg)
- [rembg PyPI](https://pypi.org/project/rembg/)
- [BiRefNet vs rembg vs U2Net comparison](https://dev.to/om_prakash_3311f8a4576605/birefnet-vs-rembg-vs-u2net-which-background-removal-model-actually-works-in-production-4830)
- [ModernGL headless documentation](https://moderngl.readthedocs.io/en/stable/techniques/headless_ubuntu_18_server.html)
- [headless-moderngl Docker project](https://github.com/JWDobken/headless-moderngl)
- [Mesa LLVMpipe Docker](https://github.com/utensils/docker-opengl)
- [colour-science PyPI](https://pypi.org/project/colour-science/)
- [colour-science Oklab module](https://github.com/colour-science/colour/blob/develop/colour/models/oklab.py)
- [colour.delta_E documentation](https://colour.readthedocs.io/en/latest/generated/colour.delta_E.html)
- [blend-modes GitHub](https://github.com/flrs/blend_modes)
- [blend-modes PyPI](https://pypi.org/project/blend-modes/)
- [vtracer GitHub](https://github.com/visioncortex/vtracer)
- [vtracer PyPI](https://pypi.org/project/vtracer/)
- [Potrace vs VTracer comparison](https://www.aisvg.app/blog/image-to-svg-converter-guide)
- [opensimplex-loops GitHub](https://github.com/Dennis-van-Gils/opensimplex-loops)
- [opensimplex PyPI](https://pypi.org/project/opensimplex/)
- [CairoSVG documentation](https://cairosvg.org/documentation/)
- [CairoSVG libcairo issue](https://github.com/Kozea/CairoSVG/issues/371)
- [Replicate pricing](https://replicate.com/pricing)
- [Replicate webhooks](https://replicate.com/docs/topics/webhooks)
- [Replicate super-resolution collection](https://replicate.com/collections/super-resolution)
- [Replicate image editing collection](https://replicate.com/collections/image-editing)
- [albumentations weather transforms](https://albumentations.ai/docs/examples/example-weather-transforms/)
- [albumentations chromatic aberration](https://albumentations.ai/docs/examples/example_chromatic_aberration/)
- [Albumentations interactive explorer](https://explore.albumentations.ai)
- [OpenCV vs Pillow comparison](https://learningdaily.dev/what-is-the-difference-between-opencv-and-pillow-457e37b7d530)
- [OpenCV vs scikit-image comparison](https://eureka.patsnap.com/article/opencv-vs-scikit-image-speed-and-functionality-tradeoffs)
- [Oklab color space](https://bottosson.github.io/posts/oklab/)
