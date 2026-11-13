set -x
P=.venv/bin/python
$P src/exp2.py --tag lgb_s0 --tseed 0 --full --novalid
for s in 1 2 3; do $P src/exp2.py --tag lgb_s$s --tseed $s --full; done
$P src/exp2.py --model cat --iters 800 --lr 0.08 --depth 7 --tag cat_s0 --tseed 0 --full --novalid
$P src/exp2.py --model cat --iters 800 --lr 0.08 --depth 7 --tag cat_s1 --tseed 1 --full
echo DONE
