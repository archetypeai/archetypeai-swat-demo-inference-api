# SWaT data prep scripts

The SWaT CSVs aren't in this repo, so run these once after downloading the raw dataset (from iTrust, or the [Kaggle mirror](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system)).

| Script                 | Purpose                                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `convert_swat_data.py` | Merge normal + attack CSVs, forward-fill missing SCADA values, drop 11 constant-value actuators, write `swat_raw_labeled.csv`                                                     |
| `generate_labels.py`   | Split the labeled timeline into n-shot training files (`swat_normal.csv`, `swat_attack.csv`) and the held-out playback stream (`swat_playback.csv`), asserting they share no rows |

Usage:

```bash
# Place SWaT_Dataset_Normal_v1.csv and SWaT_Dataset_Attack_v0.csv in data/
python3 scripts/convert_swat_data.py
python3 scripts/generate_labels.py
```

## Attribution

The SWaT dataset is the work of [iTrust, Centre for Research in Cyber Security](https://itrust.sutd.edu.sg/) at Singapore University of Technology and Design. Request the dataset through iTrust for published work.

## The other scripts

| Script                      | Purpose                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `build-scaler.js`           | Per-channel mean and standard deviation of the n-shot pool → `data/scaler.json` (committed)               |
| `build-knn-library.js`      | Embed the n-shot library with Omega 1.5 → `data/knn-library.json`                                         |
| `build-inference-sample.js` | Optional: embed held-out windows for the embedding panel's background dots → `data/inference-sample.json` |
| `eval-playback.js`          | Score the classifier on the held-out playback stream                                                      |
