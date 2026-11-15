# YouTube Recording Setup & Studio Engineering Guide
## Section 10: Recursive Self-Improvement (RSI) Video Production

**Target Audience**: Video Creators, Instructors, and Teaching Assistants  
**Deliverable Scope**: Studio Hardware, Software Configuration, OBS Studio Settings, Terminal Styling, Audio Processing, and Pre-Flight Checklist

---

## 1. Video Resolution & Display Architecture

To ensure high-definition clarity on YouTube across desktop monitors, tablets, and mobile screens, configure the display environment to **1080p (1920x1080)** or **4K UHD (3840x2160)** with a fixed 16:9 aspect ratio.

### Screen Geometry & Scaling:
- **Primary Display Resolution**: `1920 x 1080` (native) or `3840 x 2160` (scaled 200%).
- **OS Scaling**: Set Windows Display Scaling to **100%** (for 1080p) or **125%** (for 1440p) to avoid blurry text rasterization in terminal windows.
- **Refresh Rate**: 60 Hz (matches standard video stream framerate).

---

## 2. OBS Studio Master Configuration

### 2.1 Video Settings (`Settings > Video`)
- **Base (Canvas) Resolution**: `1920x1080`
- **Output (Scaled) Resolution**: `1920x1080` (or `2560x1440` for YouTube's superior VP9/AV1 codec allocation)
- **Downscale Filter**: `Lanczos (Sharpened scaling, 36 samples)`
- **Common FPS Values**: `60` (ensures ultra-smooth terminal scrolling and slide animations)

### 2.2 Output & Recording Settings (`Settings > Output > Recording`)
- **Output Mode**: `Advanced`
- **Recording Format**: `MKV` (avoids file corruption if OBS crashes; remux to `.mp4` via `File > Remux Recordings`)
- **Video Encoder**:
  - NVIDIA GPUs: `NVIDIA NVENC H.264 (new)` or `NVENC HEVC`
  - AMD GPUs: `AMD HW H.264`
  - CPU Fallback: `x264` (Preset: `medium` or `slow`)
- **Rate Control**: `CBR` (Constant Bitrate) or `CQP` (Constant Quantization Parameter)
  - For **CBR**: Set Bitrate to `16,000 kbps` (for 1080p60) or `30,000 kbps` (for 1440p60)
  - For **CQP**: Set CQ Level to `18` (high visual fidelity, mathematically near-lossless)
- **Keyframe Interval**: `2 seconds`

### 2.3 Audio Settings (`Settings > Audio`)
- **Sample Rate**: `48 kHz`
- **Channels**: `Stereo`
- **Bitrate**: `320 kbps` (AAC)

---

## 3. Audio Processing & Microphone Chain

Clear, crisp vocal audio is mandatory for educational technical walkthroughs. Apply the following real-time filter chain in OBS (`Audio Mixer > Microphone > Filters`):

1. **Noise Suppression**:
   - Method: `RNNoise (good quality, more CPU usage)`
   - Eliminates PC fan noise, air conditioner hum, and room acoustics.
2. **Noise Gate**:
   - Close Threshold: `-42 dB`
   - Open Threshold: `-35 dB`
   - Attack Time: `25 ms` | Hold Time: `200 ms` | Release Time: `150 ms`
   - Mutes ambient breathing and keyboard clicks when not speaking.
3. **Compressor**:
   - Ratio: `3.00:1`
   - Threshold: `-18 dB`
   - Attack Time: `6 ms` | Release Time: `60 ms`
   - Output Gain: `+3.0 dB`
   - Levels volume spikes when raising voice for emphasis.
4. **Limiter**:
   - Threshold: `-1.5 dB`
   - Release: `60 ms`
   - Prevents digital clipping and audio distortion.

---

## 4. Terminal Environment & Developer Aesthetics

When displaying code and CLI outputs on YouTube, visual clarity is paramount:

### 4.1 Terminal Applications
- **Windows**: Use **Windows Terminal** (v1.18+) or **VS Code Integrated Terminal**.
- **macOS / Linux**: Use **iTerm2** or **Alacritty**.

### 4.2 Color Themes & Typography
- **Theme**: High-contrast dark theme:
  - Recommended: `One Half Dark`, `Tokyo Night`, or `Catppuccin Mocha`.
  - Contrast Ratio: Minimum `7:1` between foreground text and background.
- **Font**: Monospaced font with coding ligatures:
  - Recommended: `JetBrains Mono`, `Fira Code`, or `Cascadia Code`.
  - Font Size: **18 pt** or **20 pt** (ensures legibility on mobile devices).
  - Line Height: `1.2` or `1.3`.

### 4.3 Window Geometry
- Window Width: **120 columns** minimum (prevents wrapping of wide benchmark tables).
- Window Height: **35 rows**.
- Position: Centered on screen or occupying the right 60% of the display alongside Marp presentation slides.

---

## 5. Environment Variables & Runtime Switch

Before beginning recording, configure your environment variables depending on the desired demonstration mode:

### Offline Mock Mode (Default / Deterministic Walkthrough)
Guarantees 100% reproducible numbers and sub-second runtimes matching the narration script:
```powershell
# Windows PowerShell
$env:RSI_MODE = "mock"
$env:RSI_EXECUTION_MODE = "mock"

# Linux / macOS Bash
export RSI_MODE="mock"
export RSI_EXECUTION_MODE="mock"
```

### Live Gemini API Mode (Real-Time Cloud Walkthrough)
Enables live calls to Gemini 3.8 Flash:
```powershell
# Windows PowerShell
$env:RSI_MODE = "live"
$env:GEMINI_API_KEY = "your-actual-gemini-api-key"

# Linux / macOS Bash
export RSI_MODE="live"
export GEMINI_API_KEY="your-actual-gemini-api-key"
```

---

## 6. Pre-Flight Production Checklist

Complete this checklist 15 minutes prior to hitting "Start Recording":

- [ ] **Marp Presentation Ready**: Open slide deck in browser or VS Code Marp preview (`Ctrl+K V`), press `F11` for full screen.
- [ ] **Terminal Pre-Seeded**: Navigate to `Special-Topics-in-AI/assignment-5-rsi`, activate `.venv`, test clear screen (`cls` or `clear`).
- [ ] **Audio Check**: Record a 10-second test clip speaking normally and verify waveform peaks between `-12 dB` and `-6 dB` in OBS.
- [ ] **Notifications Silenced**: Enable "Do Not Disturb" / "Focus Assist" on Windows/macOS. Close Discord, Slack, and email clients.
- [ ] **Water & Vocal Warm-up**: Keep water nearby; read the first paragraph of the script aloud to warm up vocal cords.
- [ ] **Disk Space**: Verify at least 20 GB of free storage on the recording drive.
