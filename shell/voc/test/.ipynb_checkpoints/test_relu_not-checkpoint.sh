#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../../

weight_file=./weights/VOC/relu_not_512
eval_folder=./eval/VOC/relu_not_512
for((i=10;i<=400;i+=5));do
    weight=$weight_file/relu_not_concat_VOC_epoches_$i.pth
    if [ ! -f "$weight" ]; then
        echo "weight not found!"$weight
        continue
    fi
    save_folder=$eval_folder/$i
    mkdir -p $save_folder
    echo $weight
    echo $save_folder
    python test_RFB.py -d VOC -v relu_not_concat -s 512 -r "9,5,2,1" -m  $weight --save_folder=$save_folder
done
#python eval_aspp.py --trained_model="./weights/voc_aspp_1259/VOC.pth" --save_folder="./eval/voc_aspp-1259/"
