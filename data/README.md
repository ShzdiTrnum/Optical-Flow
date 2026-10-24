# data/

Place input videos here before running the scripts. Video files are git-ignored.

## Required files

| File | Used by | Description |
|------|---------|-------------|
| `ALS - Trim20s.mp4` | `optical_flow_als_ultrasound.py` | Clinical ALS muscle ultrasound clip (~20 s). **Not distributed**, see below. |
| `fasciculation_simulation_accel_decel_test_new4.mp4` | `optical_flow_synthetic.py` | Synthetic video from the `Fasciculation-Simulation` repository |

## Data availability

The clinical ultrasound videos were recorded at Tokyo Metropolitan Neurological Hospital and are not
redistributed. To run the code without them, use the synthetic video:

1. Generate it with <https://github.com/ShzdiTrnum/Fasciculation-Simulation>.
2. Put the output `.mp4` in this folder and run `optical_flow_synthetic.py`.
