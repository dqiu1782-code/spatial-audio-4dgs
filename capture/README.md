# capture: general multi-camera / multi-mic capture for spatial-audio 4DGS

Rig spec -> clap sync -> mic self-calibration -> voice / ambience layers.

| file | role |
|---|---|
| rig.py | rig.json loader, validator, top-view layout (`python rig.py rig.json layout.svg`) |
| sync.py | 3-clap slate sync across independent recorders |
| selfcal.py | voice/background attribution + mic position/axis/latency/drift self-calibration |
| layers.py | CTF voice/ambience separation, beamformed refs, activity gate |
| sim.py | ground-truth capture simulator |
| tests/ | `test_selfcal.py rig.json [--no-lav]`, `test_layers.py rig.json [--no-lav]`, `test_replay.py` (real data) |
| examples/ | basic: 9 phones over the front 120 deg; advanced: 24 cameras around 360 deg, 8 with binaural recorders, + lavaliers |

## Camera density (picture)

Novel views are interpolated between neighbouring cameras, so the rig declares a viewing arc
(`"view": {"arc_deg": [start, end]}`, default the full ring) and puts cameras <= 15 deg apart inside it
(the validator warns above 15 deg and flags > 20 deg as blurry). About arc / 15 + 1 cameras for an arc,
24 for a full ring. On Meta Replay (7 cameras, 24-60 deg apart) the held-out view reached only 15.8 dB
even after all fixes; sharp public 4DGS results come from dense arrays (18-21 cameras or more).
Also record ~10 s with the actors off camera (`empty_plate`, can coincide with the room tone): it gives
a clean static background.

## Simulation results (audio)

40 s, 2 talkers, RT60 0.45 s:

| rig | clap sync | localisation median, nominal -> calibrated | voice removed from ambience |
|---|---|---|---|
| advanced: 24 cameras, 8 binaural + lavaliers | 0.06 ms | 29.6 -> 1.8 cm | 8.8 dB |
| basic: 9 phones + close mic (lavalier or pocket phone) | 0.07 ms | 50.2 -> 1.4 cm | 8.7 dB |
| basic: 9 phones, no close mics | - | 58.4 -> 29.3 cm (p90 113 -> 65) | 4.0 dB |

Needs numpy, scipy (plus soundfile for `tests/test_replay.py`). Simulations are cached in
`$SA4DGS_CACHE` or `capture/cache/`, keyed by the rig's acoustic content.
