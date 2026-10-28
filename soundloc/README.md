# soundloc: locating a sound source with several cameras

Idea: each camera carries a small microphone array and measures which direction the sound comes from. Camera poses are known (from COLMAP / the 4DGS reconstruction), so each direction becomes a ray in world coordinates. The rays never meet exactly, so least squares finds the point closest to all of them: that is the source position. Doing this for every time slice gives the source track, and from it directional binaural audio can be rendered for a virtual listener anywhere in the scene (listen with headphones).

## Files

| file | purpose |
|---|---|
| `simulate.py` | simulation: room + first-order wall reflections + noise, several cameras with mics recording a moving source (with ground truth) |
| `doa.py` | per-camera direction: GCC-PHAT (2 mics -> cone), SRP-PHAT (>= 3 mics -> full 3D direction) |
| `triangulate.py` | closed-form closest point to rays; robust nonlinear least squares mixing rays and cones; Kalman + RTS smoothing |
| `run_localize.py` | main program: read scene -> per-frame directions -> triangulation -> smoothing -> evaluation -> plots |
| `binaural.py` | binaural rendering for a virtual listener from the track (spherical head model: interaural time difference + head shadow + distance) |
| `make_scene_real.py` | real data: extract audio from video, read COLMAP poses, auto-sync with a clap, write a scene directory |
| `plot_result.py` | top view and error curves (needs matplotlib, skipped otherwise) |

Only numpy is required (matplotlib for plots, imageio-ffmpeg to extract audio from video).

## Run the simulation first

```
run_demo.bat
```
or by hand:
```
python simulate.py --out demo_tetra --array tetra --cams 4      # 4 cameras, 4 mics each
python run_localize.py demo_tetra                                 # localise + evaluate + plot
python binaural.py demo_tetra                                     # listener in the room centre, slowly turning
python binaural.py demo_tetra --use truth                         # render from the ground-truth track for comparison
```
Common options: `--array stereo|tetra|hex`, `--cams N`, `--spread wide|tight`, `--traj walk|circle|static`,
`--reflect 0.5` (wall reflections), `--snr 25`, `--source_wav your_sound.wav`.

## Real data

1. Record video with each camera (with microphones); clap once after starting (used for automatic time alignment).
2. Reconstruct each camera's pose with COLMAP (same frame as 3DGS/4DGS).
3. Write a config following `real_config_example.json`: each camera's video, mic layout and image name in COLMAP.
4. `python make_scene_real.py your_config.json --out real1`, then `python run_localize.py real1`.

## Simulation results (7x5x3 m room, 12 s, first-order reflections 0.5, SNR 25 dB)

| setup | per-camera direction error | localisation error (per frame / smoothed, median) |
|---|---|---|
| 4 cameras spread out, 4 mics each | ~1 deg | 0.08 m / 0.07 m |
| 4 cameras spread out, 2 mics each (cone only) | angle ~0.5 deg | 0.43 m / 0.22 m |
| 4 cameras clustered, 4 mics each | ~1.2 deg | 0.56 m / 0.51 m |

Spreading the cameras matters more than adding microphones: when the cameras are close together the rays are nearly parallel and the intersection is poorly constrained.
