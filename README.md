# UPI Fraud Rings — ML Gradient Wars 2026

12-class per-transaction fraud labelling (macro-F1). Code for our submissions.

## Setup
```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python pandas pyarrow numpy numba lightgbm catboost scikit-learn
# put the competition files in data/upi-fraud-rings/ (train.csv, test.csv, accounts.csv, merchants.csv)
```

## Reproduce the best public upload (`v13_oofu`, public 0.5124)
Built with features v3 = git commit `559e3a3`. Exact commands: `docs/experiments.md` row #37.
```
python src/connectivity.py && python src/connectivity.py --until 2026-09-01   # label-structure flags (data/conn*.parquet)
python src/oof.py && python src/relabel.py                                    # data/relabel_R1.parquet
python src/train_full.py --name v6_R1_bag --clean iso7 --relabel data/relabel_R1.parquet --seeds 0,1,2,3 --iters 400 --drop ip_rank_month
python src/bag_val.py --seeds 1,2,3 --relabel data/relabel_R1.parquet ...      # validation probs for class weights
python src/make_sub.py --pred preds/v6_R1_bag_test_P.npy --val_P /tmp/P_bag4_R1.npy --wmode full --out v6_R1_bag   # public 0.4685
python src/oct_rules.py --base submissions/v6_R1_bag.csv --rules QAABCD --out v10_v6_octrules                      # public 0.4894
python src/make_blockseg.py --base submissions/v10_v6_octrules.csv --out submissions/v11_blockseg.csv               # public 0.5044
python src/make_tailfix.py --base submissions/v11_blockseg.csv --out submissions/v12_tailfix.csv                # public 0.5117
```
`oct_rules.py` = October rules (QR->P2P blank mismatch, device-ID fraud block); `make_blockseg.py` re-types the October fraud
device block by ordered segmentation (`src/block_seg.py`). Both are coded rules on provided columns; no hand labels.

v13 = same rules on a cleaner base model (bag of 4 LightGBM seeds trained with `--clean iso7oofu`):
```
python src/oof.py full                                   # full-frame week-blocked OOF -> /tmp/oof_sfull.npy
python - <<'PY'                                          # drop list: fraud rows with OOF P(legit)>0.8 + legit rows with OOF P(legit)<0.1
import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd
from features import load, sample_frame
tr,te,_,_=load(); df=sample_frame(tr,te,0,rate=1.0); O=np.load('/tmp/oof_sfull.npy'); lab=df.label.values; t=(df._is_test==0).values
bad=(t&(lab!='legit')&(O[:,0]>0.8))|(t&(lab=='legit')&(O[:,0]<0.1)); pd.DataFrame({'txn_id':df.txn_id.values[bad]}).to_parquet('data/oofbadfull_u.parquet')
PY
for s in 0 1 2 3; do python src/exp2.py --model lgb --tseed $s --clean iso7oofu --full --tag oofu_f$s; done
python - <<'PY'                                          # average the 4 seeds (val + test)
import numpy as np, pandas as pd
np.save('/tmp/P_oofu_bag.npy', np.mean([np.load(f'/tmp/V_oofu_f{s}.npy') for s in range(4)],0)); np.save('/tmp/P_oofu_bag_ids.npy', np.load('/tmp/V_oofu_f0_ids.npy',allow_pickle=True))
np.save('preds/oofu_bag_test_P.npy', np.mean([np.load(f'preds/oofu_f{s}_test_P.npy') for s in range(4)],0))
pd.read_csv('preds/oofu_f0_test_ids.csv').to_csv('preds/oofu_bag_test_ids.csv', index=False)
PY
python src/make_sub.py --pred preds/oofu_bag_test_P.npy --val_P /tmp/P_oofu_bag.npy --wmode full --out v13base_oofu
python src/oct_rules.py --base submissions/v13base_oofu.csv --rules QAABCD --out v13r_oofu
python src/make_blockseg.py --base submissions/v13r_oofu.csv --out submissions/v13s_oofu.csv
python src/make_tailfix.py --base submissions/v13s_oofu.csv --out submissions/v13_oofu.csv      # public 0.5124
```

## Layout
- `src/features.py` — features on a 51%-density frame (test is a ~51% sample of October); `src/common.py` — metric, cleaning, relabel.
- `src/exp.py`, `src/exp2.py` — validation (Jul–Aug → Sep); `src/train_full.py` — full training; `src/make_sub.py` — CSV.
- `docs/experiments.md` — every experiment and LB score; `docs/insights.md` — verified data facts.
