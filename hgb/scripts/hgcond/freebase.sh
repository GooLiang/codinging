#!/bin/bash
python /home/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.1 --pr 0.85 --ACM-keep-F True
python /home/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.2 --pr 0.85 --ACM-keep-F True 
python /home/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.3 --pr 0.85 --ACM-keep-F True 
python /home/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.4 --pr 0.85 --ACM-keep-F True 
python /home/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.5 --pr 0.85 --ACM-keep-F True