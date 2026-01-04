# Qwen-Poster-Studio

## Qwen Poster Studio (VL → Brief → Qwen-Image Poster)

A lightweight **Gradio** app that turns **rough poster drafts + product inputs** into a **high-converting e-commerce poster** using a two-stage pipeline:

1) **Qwen2.5-VL-3B-Instruct (4-bit)** analyzes an uploaded draft and produces a structured **design brief** (layout, typography, palette, improvements).  
2) **Qwen-Image-2512** generates a final poster using your product details + style keywords + the draft-based brief.

Designed to run smoothly on an **A100** (recommended). It can run on smaller GPUs if you **disable the draft-brief step**, use **smaller resolutions**, and keep **steps** modest.

---

## Demo Features

- **Draft-to-brief (optional):** Upload an early mock/poster draft → model returns a clean design brief.
- **Poster generation:** Enter product name/description/offer/price/CTA/benefits + tone/style → generate a promo poster.
- **Platform presets:** One-click aspect ratios/sizes for Instagram posts/stories, banners, posters, slides.
- **Memory-aware:** VL model loads **only inside** the describe function, then unloads.

---

## Pipeline Overview

### Stage A — Draft → Design Brief (Vision-Language)

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Loading mode:** 4-bit quantization via `bitsandbytes`
- **When used:** Only if `use_draft_brief=True` and a draft image is provided.

**Flow**
1. User uploads a draft image (**Gradio → numpy**).
2. Converted to **PIL** and resized with `thumbnail((1024, 1024))` to reduce overhead.
3. Prompt instructs the VL model to output **exact sections**:
   - Layout, Style, Color palette, Typography, Keep, Improve, Notes
4. VL output becomes a **human-readable design brief**.
5. VL model is deleted and memory is cleaned (`del`, `gc.collect()`, `torch.cuda.empty_cache()`).

**Why this design?**  
The VL step is heavy. Loading it only when needed keeps the app responsive and reduces VRAM spikes.

---

### Stage B — Product Info + Brief → Final Prompt

A deterministic prompt builder merges:
- Product fields (name/description/offer/price/CTA/benefits)
- Creative controls (tone/style keywords/language)
- Optional draft brief block

This produces one “poster spec” prompt sent to Qwen-Image.

---

### Stage C — Text-to-Image Poster Generation

- **Model:** `Qwen/Qwen-Image-2512`
- **Diffusers pipeline:** `QwenImagePipeline`

**Stability + performance choices (A100-friendly)**
- `DTYPE = torch.bfloat16` on GPU
- **VAE forced to FP32** (`t2i.vae.to(torch.float32)`) to reduce NaN/black-image decode issues
- `vae.enable_tiling()` to reduce peak memory at higher resolutions
- `low_cpu_mem_usage=True` + `safetensors` for lighter CPU RAM behavior during load

Output returns to Gradio as the final poster image.

---

## Requirements

### Recommended Hardware
- **A100 (80GB):** Best experience; supports hi-res presets like `1328×1328` with more steps.
- **T4 (16GB):** Works for smaller sizes (e.g., 768 presets) and typically with `use_draft_brief=False`.

### Software
- Python (Google Colab works great)
- CUDA GPU runtime
- Libraries: `diffusers`, `transformers`, `accelerate`, `bitsandbytes`, `gradio`, `qwen-vl-utils`, `pillow`, `psutil`

---

## Installation (Colab)

```bash
!pip -q install --upgrade pip
!pip -q install git+https://github.com/huggingface/diffusers
!pip -q install git+https://github.com/huggingface/transformers accelerate
!pip -q install gradio pillow safetensors sentencepiece bitsandbytes qwen-vl-utils psutil
```
---
## Run

Run the Python script / notebook cell that contains the app code.
Gradio will launch a shareable link:

```bash
demo.launch(share=True, debug=True)
```
