# ImViD Opera — Demo Results

## Binaural Audio Demo

| file | description |
|------|-------------|
| `imvid_opera_binaural.wav` | HRTF-rendered binaural audio from a virtual listener at cam00 (3 m from singer), 48 kHz stereo, 5.2 s |
| `imvid_opera_compare_audio.wav` | A/B comparison: left channel = real cam00 recording, right channel = HRTF render |

Listen with headphones. The binaural render uses MIT KEMAR HRIR database with direct-path propagation delay (8.7 ms), 1/r attenuation, and directional filtering for the singer's position.

## 4D Gaussian Splatting Visuals

| file | description |
|------|-------------|
| `imvid_opera_render.gif` | 4DGS render sequence (50 frames, 640 px) |
| `imvid_opera_compare.gif` | Ground truth vs render side-by-side comparison |
| `imvid_opera_sample.png` | Single sample frame |

Trained on 38 cameras × 50 frames from the ImViD opera dataset (Tsinghua, CVPR 2025, CC BY 4.0). Held-out camera achieves 25.8 dB PSNR.

## Dataset

[ImViD: Immersive Video Database](https://www.volic.cs.cmu.edu/imvid/) — 39 GoPros on a ~1.2 m spherical rig, opera singer, 60 fps, stereo audio per camera.
