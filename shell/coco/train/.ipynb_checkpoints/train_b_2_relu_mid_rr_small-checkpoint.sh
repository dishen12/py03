#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../../

python train_RFB.py -d COCO -v relu_mid_rr -s 512 -r "5,3,2,1" --save_folder="./weights/coco/relu_mid_rr_512_small/" --ngpu=4