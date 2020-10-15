#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

python train.py --diff $1 --diff-type random --lr $3 --epochs 1000 --mom 0.9 --seed $2 --save-ntk-train --save-ntk-test --layer-align-train --layer-align-test --align-train --align-test --task cifar10_vgg19
