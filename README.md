# braindance-audio

Spatial audio for multi-camera volumetric video ("Braindance"): capture spec, clap sync, microphone self-calibration, voice / ambience layering, multi-camera sound-source localisation, and HRTF binaural rendering, plus a viewer that plays a [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians) scene with listener-relative binaural audio. Tested in simulation and on real data ([Meta Replay](https://github.com/facebookresearch/Replay), [ImViD](https://github.com/Metaverse-AI-Lab-THU/ImViD)).

The goal: when you walk around inside a reconstructed 3D video, the sound should come from where the person actually is, and change as you move and turn your head.

## What is here

```
capture/     general capture workflow (basic tier: phones; advanced tier: DSLRs + external mics)
  rig.py       read and validate rig.json (camera angle / height / orientation rules), top-view layout SVG
  sync.py      three-clap sync (finds 3 strong, evenly spaced onsets, so a door knock is not mistaken for a clap)
  selfcal.py   decides which actor is speaking in each window, then self-calibrates every mic's position, orientation, delay and drift
  layers.py    CTF voice / ambience layering, beamformed reference, activity gate
  sim.py       capture simulator with ground truth
  scale.py     metric scale for a COLMAP model from measured distances between a few cameras; also measures distances between two image points
  tests/       test_selfcal / test_layers (simulation), test_replay (real Replay data)
  examples/    basic: 9 phones over 120 deg in front; advanced: 24 cameras all around (8 with binaural recorders) + lavalier mics
soundloc/    multi-camera sound localisation: per-camera direction (GCC/SRP-PHAT) -> ray triangulation -> Kalman smoothing -> binaural rendering
replay/      Meta Replay tools: download, metadata, cameras (normalised world frame), audio-video sync, audio-only localisation, HRTF renderer
imvid/       the full pipeline on the ImViD opera data (39 GoPros, each with audio): frame prep, scale, sync, localisation, voice, 4DGS training
viewer/      Braindance viewer: 4DGaussians rendering + HRTF binaural audio that follows the listener's position and heading
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

Simulation results are cached in `$BRAINDANCE_CACHE`, or `capture/cache/` if it is not set.

## Real data: Meta Replay

The Replay dataset (Meta; CC BY-NC 4.0; download it yourself and follow its license) is expected in `E:\replay` by default; override with the `REPLAY_ROOT` environment variable.

```bash
python replay/replay_fetch.py SC-1044        # download one scene (8 DSLR videos + all audio)
python replay/replay_sync.py SC-1044         # audio-video sync, triangulate actors' heads -> sync.json
python capture/tests/test_replay.py SC-1044  # calibrate on the first half, test on the second half
```

## Viewer

Needs Windows/Linux + an NVIDIA GPU, [4DGaussians](https://github.com/hustvl/4DGaussians) (with mmcv), [COLMAP](https://colmap.github.io), pygame, sounddevice,
soundfile, and the [MIT KEMAR HRIR database](https://mcgill.ca/mpl/kemar-database) (`replay/kemar_fetch.py`). See `imvid/run_imvid.bat` for usage.

- Drag the mouse to turn your head, WASD to walk, Q/E down/up, Shift to go faster, Space to pause, Left/Right arrows to seek, 1-9 to jump to a training camera, R to reset.
- B: switch to the real recording of the camera nearest to you, for an A/B comparison with the rendered binaural audio.
- The viewer turns off the system input method so WASD works even when a CJK input method is active.

4DGaussians and its dependency diff-gaussian-rasterization are not included and must be installed separately; the latter's license allows non-commercial research use only.

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

## License

MIT, see [LICENSE](LICENSE). Datasets and third-party code mentioned above keep their own licenses.
