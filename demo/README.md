# ImViD Opera — Demo Results

## Binaural Audio Demo

| file | description |
|------|-------------|
| `imvid_opera_binaural.wav` | HRTF-rendered binaural audio from a virtual listener at cam00 (3 m from singer), 48 kHz stereo, 5.2 s |
| `imvid_opera_compare_audio.wav` | A/B comparison: left channel = real cam00 recording, right channel = HRTF render |

Listen with headphones. The binaural render uses the MIT KEMAR HRIRs (Gardner & Martin, 1995) with direct-path propagation delay (8.7 ms), 1/r attenuation, and directional filtering for the singer's position.

## 4D Gaussian Splatting Visuals

| file | description |
|------|-------------|
| `imvid_opera_render.gif` | 4DGS render sequence (50 frames, 640 px) |
| `imvid_opera_compare.gif` | Ground truth vs render side-by-side comparison |
| `imvid_opera_sample.png` | Single sample frame |

Trained on 38 cameras × 50 frames from the ImViD opera sample. Held-out camera achieves 25.8 dB PSNR.

## Source and license

These files are adapted from the opera sample (`scene1_opera.zip`) of **ImViD**: Z. Yang et al., *ImViD: Immersive Volumetric Videos for Enhanced VR Engagement*, CVPR 2025 ([GitHub](https://github.com/Metaverse-AI-Lab-THU/ImViD)). 39 GoPros on a ~1.2 m spherical rig, 60 fps, stereo audio per camera. The original is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Changes made: the "ground truth" frames and the cam00 audio come from the original (frames resized); the 4DGS frames are renders of a model trained on the original frames; the binaural track is a new rendering of the singer's voice extracted from the original recordings. These files are shared under CC BY 4.0 (not the MIT license that covers the code).
