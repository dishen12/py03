#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../../

python train_RFB.py -d VOC -v relu_not_concat -s 512 -r "9,5,2,1" --save_folder="./weights/VOC/relu_not_512/" --ngpu=4