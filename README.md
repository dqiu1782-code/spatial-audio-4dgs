# spatial-audio-4dgs

Spatial audio for multi-camera volumetric video and 4D Gaussian Splatting (4DGS): capture spec, clap sync, microphone self-calibration, voice / ambience layering, multi-camera sound-source localisation, and HRTF binaural rendering, plus a viewer that plays a [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians) [4] scene with listener-relative binaural audio. Tested in simulation and on real data ([Meta Replay](https://github.com/facebookresearch/Replay) [1], [ImViD](https://github.com/Metaverse-AI-Lab-THU/ImViD) [2]).

The goal: when you walk around inside a reconstructed 3D video, the sound should come from where the person actually is, and change as you move and turn your head.

## What is here

```
capture/     general capture workflow (basic tier: phones; advanced tier: DSLRs + external mics)
  rig.py       read and validate rig.json (camera angle / height / orientation rules), top-view layout SVG
  sync.py      three-clap sync (finds 3 strong, evenly spaced onsets, so a door knock is not mistaken for a clap)
  selfcal.py   decides which actor is speaking in each window, then self-calibrates every mic's position, orientation, delay and drift
  layers.py    CTF [13] voice / ambience layering, beamformed reference, activity gate
  sim.py       capture simulator with ground truth
  scale.py     metric scale for a COLMAP model from measured distances between a few cameras; also measures distances between two image points
  tests/       test_selfcal / test_layers (simulation), test_replay (real Replay data)
  examples/    basic: 9 phones over 120 deg in front; advanced: 24 cameras all around (8 with binaural recorders) + lavalier mics
soundloc/    multi-camera sound localisation: per-camera direction (GCC-PHAT [8] / SRP-PHAT [9]) -> ray triangulation -> Kalman + RTS smoothing [12] -> binaural rendering [10, 11]
replay/      Meta Replay tools: download, metadata, cameras (normalised world frame), audio-video sync, audio-only localisation, HRTF renderer
imvid/       the full pipeline on the ImViD opera data (39 GoPros, each with audio): frame prep, scale, sync, localisation, voice, 4DGS training
viewer/      spatial_viewer.py: 4DGaussians rendering + HRTF binaural audio that follows the listener's position and heading
demo/        Pre-rendered ImViD opera results: binaural audio demo, 4DGS visuals, A/B comparison
```

## Quick start

Requirements: Python 3, numpy, scipy; `soundfile` for real audio; `remotezip` to download Replay; `matplotlib` optional (plots).

```bash
# simulation: multi-camera sound localisation
cd soundloc
python simulate.py --out demo_tetra --array tetra --cams 4
python run_localize.py demo_tetra
python binaural.py demo_tetra

# capture workflow: validate a rig, draw its layout, then test self-calibration and layering in simulation
cd capture
python rig.py examples/rig_advanced_binaural.json layout.svg
python tests/test_selfcal.py examples/rig_advanced_binaural.json
python tests/test_layers.py examples/rig_advanced_binaural.json
```

Simulation results are cached in `$SA4DGS_CACHE`, or `capture/cache/` if it is not set.

## Real data: Meta Replay

The Replay dataset (Meta; download it yourself and follow its license) is expected in `E:\replay` by default; override with the `REPLAY_ROOT` environment variable.

```bash
python replay/replay_fetch.py SC-1044        # download one scene (8 DSLR videos + all audio)
python replay/replay_sync.py SC-1044         # audio-video sync, triangulate actors' heads -> sync.json
python capture/tests/test_replay.py SC-1044  # calibrate on the first half, test on the second half
```

## Viewer

Needs Windows/Linux + an NVIDIA GPU, [4DGaussians](https://github.com/hustvl/4DGaussians) (with mmcv), [COLMAP](https://colmap.github.io) [6], pygame, sounddevice,
soundfile, and the [MIT KEMAR HRIRs](https://sound.media.mit.edu/resources/KEMAR/hrtfdoc.txt) [7] (downloaded by `replay/kemar_fetch.py`). See `imvid/run_imvid.bat` for usage.

- Drag the mouse to turn your head, WASD to walk, Q/E down/up, Shift to go faster, Space to pause, Left/Right arrows to seek, 1-9 to jump to a training camera, R to reset.
- B: switch to the real recording of the camera nearest to you, for an A/B comparison with the rendered binaural audio.
- The viewer turns off the system input method so WASD works even when a CJK input method is active.

4DGaussians and its dependency diff-gaussian-rasterization (from 3D Gaussian Splatting [5]) are not included and must be installed separately; the latter's license allows non-commercial research use only. The viewer imports them at run time and contains no code copied from them.

## Main results

ImViD opera (39 GoPros, see `imvid/README.md`): held-out view PSNR 25.8 dB. After syncing on the singer's voice onset, audio-only localisation is 11 cm from the mouth position tracked in the video (160 cm without that sync).

**Try it:** Listen to the [binaural audio demo](demo/imvid_opera_binaural.wav) (headphones) — singer's dry signal rendered with KEMAR HRTF from a virtual listener 3 m away. The [A/B comparison](demo/imvid_opera_compare_audio.wav) plays the real cam00 recording (left) vs. HRTF render (right). See also [render GIF](demo/imvid_opera_render.gif) and [GT vs render comparison](demo/imvid_opera_compare.gif).

Camera spacing rule: adjacent cameras no more than 15 deg apart within the range of allowed viewing directions. On Replay (7 cameras, 24-60 deg apart) a view not used in training reached only 15.8 dB even after every improvement I tried, so camera density is a precondition for a sharp picture.

Audio in simulation: 40 s, 2 actors, RT60 0.45 s.

| setup | localisation median (before -> after calibration) | voice suppression in the ambience layer |
|---|---|---|
| advanced: 24 cameras, 8 with binaural recorders, + lavaliers | 29.6 -> 1.8 cm | 8.8 dB |
| basic: 9 phones + close mic (lavalier or a phone in a pocket) | 50.2 -> 1.4 cm | 8.7 dB |
| basic: 9 phones, no close mic | 58.4 -> 29.3 cm | 4.0 dB |

Replay SC-1044: 8 DSLRs, each with a 3Dio binaural recorder; mic positions, orientations and delays all unknown.

- Calibrating on the first half and testing on the second: horizontal localisation 6.7 cm, delay prediction error 0.16 ms.
- Using only the metadata without calibration, not a single window converges.
- Recorder delays range from -26 to +59 ms and are stable to within +-2 ms across runs.
- Voice / ambience layering: the ambience layer drops 6.6 dB while someone speaks, and only 0.7 dB during pauses.

## Data and third-party software

No dataset files, model weights or third-party source code are included, except the demo media in `demo/`, which are derived from ImViD (see below). The scripts download or read everything else from where you put it, and its own terms apply.

| used for | what | license / terms |
|---|---|---|
| real-data tests of calibration, layering and localisation | Meta Replay [1] (scene SC-1044) | see the terms on the Replay release page; download it yourself |
| full audio-visual pipeline and the media in `demo/` | ImViD [2], `scene1_opera.zip` sample | CC BY 4.0 |
| 4DGS sanity check (`imvid/colmap_known_poses.py`) | Neural 3D Video (N3DV / DyNeRF) [3] | CC BY-NC 4.0 |
| dynamic scene reconstruction and viewer rendering | 4DGaussians [4], diff-gaussian-rasterization from 3DGS [5] | see their repositories (rasteriser: non-commercial research only) |
| binaural rendering | MIT KEMAR HRTF measurements [7] | freely available from MIT Media Lab; please cite [7] |
| person detection for metric scale | YOLOX-s [14] (ONNX weights, run with OpenCV `cv2.dnn`) | Apache-2.0 |
| initial point clouds from known poses | COLMAP [6] | BSD |
| general | NumPy, SciPy, PyTorch, OpenCV, FFmpeg, pygame, sounddevice, soundfile, Matplotlib | see each package |

## References

1. R. Shapovalov et al. Replay: multi-modal multi-view acted videos for casual holography. *ICCV*, 2023.
2. Z. Yang et al. ImViD: immersive volumetric videos for enhanced VR engagement. *CVPR*, 2025.
3. T. Li et al. Neural 3D video synthesis from multi-view video. *CVPR*, 2022.
4. G. Wu et al. 4D Gaussian splatting for real-time dynamic scene rendering. *CVPR*, 2024.
5. B. Kerbl, G. Kopanas, T. Leimkühler and G. Drettakis. 3D Gaussian splatting for real-time radiance field rendering. *ACM Transactions on Graphics* 42(4), 2023.
6. J. L. Schönberger and J.-M. Frahm. Structure-from-motion revisited. *CVPR*, 2016. J. L. Schönberger, E. Zheng, M. Pollefeys and J.-M. Frahm. Pixelwise view selection for unstructured multi-view stereo. *ECCV*, 2016.
7. W. G. Gardner and K. D. Martin. HRTF measurements of a KEMAR. *Journal of the Acoustical Society of America* 97(6), 1995.
8. C. H. Knapp and G. C. Carter. The generalized correlation method for estimation of time delay. *IEEE Transactions on Acoustics, Speech, and Signal Processing* 24(4), 1976.
9. J. H. DiBiase, H. F. Silverman and M. S. Brandstein. Robust localization in reverberant rooms. In *Microphone Arrays*, Springer, 2001.
10. C. P. Brown and R. O. Duda. A structural model for binaural sound synthesis. *IEEE Transactions on Speech and Audio Processing* 6(5), 1998.
11. J. B. Allen and D. A. Berkley. Image method for efficiently simulating small-room acoustics. *Journal of the Acoustical Society of America* 65(4), 1979.
12. H. E. Rauch, F. Tung and C. T. Striebel. Maximum likelihood estimates of linear dynamic systems. *AIAA Journal* 3(8), 1965.
13. R. Talmon, I. Cohen and S. Gannot. Relative transfer function identification using convolutive transfer function approximation. *IEEE Transactions on Audio, Speech, and Language Processing* 17(4), 2009.
14. Z. Ge, S. Liu, F. Wang, Z. Li and J. Sun. YOLOX: exceeding YOLO series in 2021. arXiv:2107.08430, 2021.

## License

Code: MIT, see [LICENSE](LICENSE). The media in `demo/` are adapted from ImViD [2] and are shared under CC BY 4.0, not MIT. Datasets and third-party code mentioned above keep their own licenses.
