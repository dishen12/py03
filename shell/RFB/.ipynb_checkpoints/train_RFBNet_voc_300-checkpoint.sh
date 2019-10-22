#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../

python train_old.py -d VOC -v RFB_vgg  -s 300 -max 300 --basenet="./weights/pretrain/vgg16_reducedfc.pth" --save_folder="./weights/voc/rfb_300/" --ngpu=1