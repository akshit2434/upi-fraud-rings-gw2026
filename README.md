# UPI Fraud Rings — ML Gradient Wars 2026

12-class per-transaction fraud labelling (macro-F1). Code for our submissions.

## Setup
```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python pandas pyarrow numpy numba lightgbm catboost scikit-learn
# put the competition files in data/upi-fraud-rings/ (train.csv, test.csv, accounts.csv, merchants.csv)
```

## Reproduce the best public upload (`v11_blockseg`, public 0.5044)
Built with features v3 = git commit `559e3a3`. Exact commands: `docs/experiments.md` row #37.
```
python src/connectivity.py && python src/connectivity.py --until 2026-09-01   # label-structure flags (data/conn*.parquet)
python src/oof.py && python src/relabel.py                                    # data/relabel_R1.parquet
python src/train_full.py --name v6_R1_bag --clean iso7 --relabel data/relabel_R1.parquet --seeds 0,1,2,3 --iters 400 --drop ip_rank_month
python src/bag_val.py --seeds 1,2,3 --relabel data/relabel_R1.parquet ...      # validation probs for class weights
python src/make_sub.py --pred preds/v6_R1_bag_test_P.npy --val_P /tmp/P_bag4_R1.npy --wmode full --out v6_R1_bag   # public 0.4685
python src/oct_rules.py --base submissions/v6_R1_bag.csv --rules QAABCD --out v10_v6_octrules                      # public 0.4894
python src/make_blockseg.py --base submissions/v10_v6_octrules.csv --out submissions/v11_blockseg.csv               # public 0.5044
```
`oct_rules.py` = October rules (QR->P2P blank mismatch, device-ID fraud block); `make_blockseg.py` re-types the October fraud
device block by ordered segmentation (`src/block_seg.py`). Both are coded rules on provided columns; no hand labels.

## Layout
- `src/features.py` — features on a 51%-density frame (test is a ~51% sample of October); `src/common.py` — metric, cleaning, relabel.
- `src/exp.py`, `src/exp2.py` — validation (Jul–Aug → Sep); `src/train_full.py` — full training; `src/make_sub.py` — CSV.
- `docs/experiments.md` — every experiment and LB score; `docs/insights.md` — verified data facts.
