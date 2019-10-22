#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../../

python train_RFB.py -d COCO -v relu_mid_rr -s 512 -r "9,5,2,1" --save_folder="./weights/coco/relu_mid_rr_512_mid/" --ngpu=4