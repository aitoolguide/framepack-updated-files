# 🎬 Framepack Optimized for 6GB VRAM (Pinokio Setup)

This repository contains optimized code files and instructions for running the **Framepack video generation model** on a **6GB VRAM GPU**, using the **Pinokio launcher**.

The modifications shown here were demonstrated in my [YouTube tutorial](#) where I walk through installation, setup, and real-time video generation tests.

---

## 📌 Features

- ✅ Optimized for RTX 3000 / 6GB VRAM GPUs  
- ✅ Works with Pinokio GUI (no manual terminal setup required)  
- ✅ Resolution support for `320x224` and `480x272`  
- ✅ Improved memory cleanup and NaN handling  
- ✅ Reduced generation time (from 1.5 hours to ~25–40 minutes for 3s video)

---

## 📂 Updated Files in This Repo

The following files have been modified from the original Framepack codebase:

- `app/demo_gradio.py`  
- `app/diffusers_helper/bucket_tools.py`  
- `app/diffusers_helper/pipelines/k_diffusion_hunyuan.py`  
- `app/diffusers_helper/models/hunyuan_video_packed.py`

Each file includes inline comments to explain the changes made.

---

## 📦 How to Use

### 1. Install Framepack via Pinokio

1. Download and install Pinokio: https://pinokio.computer  
2. Search and install the **Framepack** app from the Pinokio library  
3. Launch the Framepack UI using the Pinokio interface

### 2. Replace Modified Files

1. Download the files from this repository  
2. Navigate to your local Framepack directory inside Pinokio  
   Example path:  
   `C:/pinokio/api/Frame-Pack.git/app/`
3. Replace the original files with the updated versions from this repo  
4. Restart Framepack from Pinokio

> 🔧 If you're a beginner, you can skip editing the code and **just replace the files** as shown in the [tutorial video](#).

---

## ▶️ Watch the Tutorial

📺 [YouTube Video: How to Run Framepack on 6GB GPU Using Pinokio](#)

The video covers:
- What is Pinokio and why it’s useful
- Installing Framepack via Pinokio and manually
- How to replace files from this repo
- Common issues and solutions
- Live GPU utilization and stopwatch demo
- Generation examples and results

---

## 🙌 Credits

- Framepack: [Tencent ARC](https://github.com/TencentARC/FramePack)
- Pinokio Automation Tool: [pinokio.computer](https://pinokio.computer)

---

## 📝 License

This repo is shared for educational and community support purposes. Original model and code credits go to Tencent ARC.

---

👍 If this project helped you, feel free to **star** this repo and **subscribe** to my channel for more tutorials!
