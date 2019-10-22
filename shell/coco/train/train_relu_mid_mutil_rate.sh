#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../../

python train_RFB.py -d COCO -v relu_mid_mutil_rate -s 512 -r "9,5,2,1,6,3,2,1,9,5,2,1" -b 96 -max 400 --save_folder="./weights/coco/relu_mid_mutil_rate_512/" --ngpu=4