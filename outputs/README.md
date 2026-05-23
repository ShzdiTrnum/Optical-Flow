# outputs/

Generated files are saved here after running either optical flow script.

| File | Script | Description |
|------|--------|-------------|
| `Normalized Rotation Values ... .csv` | ALS script | Per-ring rotation values, frames 6–24 |
| `Normalized_Rotation_Values_Frames_0_40.csv` | Synthetic script | Per-ring rotation values, frames 0–40 |
| `rotation_plot_als.png` | ALS script | Rotation trajectory plot |
| `rotation_plot_synthetic.png` | Synthetic script | Rotation trajectory plot |

> The synthetic CSV (`Normalized_Rotation_Values_Frames_0_40.csv`) is used as
> `Original_Groundtruth_Values.csv` in the `Fasciculation-Detection-Using-Synthetic-Data` repository.

> Output files are excluded from version control via `.gitignore`.
