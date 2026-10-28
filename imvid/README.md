# imvid: full pipeline on real multi-camera audio-visual data

ImViD Opera sample (Tsinghua, CVPR 2025, CC BY 4.0; `scene1_opera.zip` from the
[ImViD GitHub release](https://github.com/Metaverse-AI-Lab-THU/ImViD/releases)):
39 GoPros on a ~1.2 m spherical rig, 5 s at 60 fps, stereo audio on every camera.
The dataset is not included here.

| step | script | where |
|---|---|---|
| undistort, 960x540 frames at 10 fps, cam00 held out, Blender-style transforms | `imvid_prepare.py SRC OUT` | any machine (needs ffmpeg) |
| person boxes (YOLOX-s ONNX via cv2.dnn), head/feet triangulation, metric scale from standing height | `imvid_track.py OUT yolox_s.onnx` | any |
| move to metres, origin under the singer, z up | `imvid_finalize.py OUT` | any |
| sound: voice-onset slate sync, SRP-PHAT localisation, beamformed dry voice, viewer sound scene | `imvid_audio.py OUT AUDIO_DIR` | any |
| COLMAP init cloud from known poses, 4DGaussians training, held-out PSNR | `run_imvid.ps1` | Windows + CUDA |
| play: 4DGS + HRTF binaural audio that follows the listener | `run_imvid.bat` | Windows + CUDA |

`AUDIO_DIR` holds `camXX.wav` / `camXX.flac` extracted from the videos
(`ffmpeg -i camXX.mp4 -vn camXX.wav`).

## Results

- Held-out camera cam00: **25.8 dB** PSNR (train 33.9 dB), 4DGaussians 14k iterations with a deformation mask.
- Sync: no clap was recorded, so the singer's voice attack (2.476 s) is the slate; propagation from the
  mouth (video track) is removed per recorder. Offsets -38..+34 ms (one at -80 ms); cam27 had no clear attack and is dropped.
- Audio-only localisation (38 recorders, SRP-PHAT, 0.25 s windows) vs the video mouth track:
  **11 cm** median (direction 0.2 deg). Without the slate sync: 160 cm.
  The sync itself uses the mouth position at the onset and the singer barely moves, so this is a consistency
  check, not an independent test.
- Beamformed dry voice: about 5 dB less low-frequency room noise than the nearest single mic.

## Limits

- The rig faces outwards: only 14 cameras see the singer, from a 16 deg arc about 3 m away. Views from her
  side are blurry because no camera saw them (see the camera-density rule in `capture/README.md`).
- Metric scale assumes a 1.62 m standing height (about 5-10 % error). With a real rig, measure two or three
  camera distances and use `capture/scale.py`.
