#!/bin/bash
python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.9 --pr 0.85 --ACM-keep-F True
#1.31
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method kcenter --num-hops 2 --reduction-rate 0.003
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method kcenter --num-hops 2 --reduction-rate 0.001
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method HGcond --num-hops 2 --reduction-rate 0.003
#1.29
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method kcenter --reduction-rate 0.3 --pr 0.85 --ACM-keep-F True
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.1
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.15
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.2
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.25
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.3
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.35
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.4
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.45
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.5
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.55
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.6
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.65
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.7
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.75
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.8
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.85
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.9
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 2 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method HGcond --reduction-rate 0.3 --alpha 0.95


















# 116
#!/bin/bash
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method HGcond --num-hops 2 --reduction-rate 0.003
##1.30
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method herding --num-hops 2 --reduction-rate 0.003
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method herding --num-hops 2 --reduction-rate 0.001
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --method herding --num-hops 2 --reduction-rate 0.002

#1.29
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.1
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.2
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.25
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.3
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.35
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.4
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.45
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.5
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.55
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.6
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.65
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.7
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.75
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.8
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.85
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.9
# python /home/public/lyx/HGcond/ogbn/train_ogbn.py --dataset ogbn-mag --num-hops 2 --method HGcond --alpha 0.95
# python /home/public/lyx/HGcond/hgb/train_hgb.py --dataset Freebase --num-hops 1 --num-hidden 512 --lr 0.001 --dropout 0.5 --eval-every 1 --num-epochs 400 --residual --SGA --method herding --reduction-rate 0.3 --pr 0.85 --ACM-keep-F True


