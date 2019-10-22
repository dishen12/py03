#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../../

weight_file=./weights/coco/relu_mid_mutil_rate_512
eval_folder=./eval/coco/relu_mid_mutil_rate_512
for((i=255;i<=400;i+=5));do
    weight=$weight_file/relu_mid_mutil_rate_COCO_epoches_$i.pth
    if [ ! -f "$weight" ]; then
        echo "weight not found!"$weight
        continue
    fi
    save_folder=$eval_folder/$i
    mkdir -p $save_folder
    echo $weight
    echo $save_folder
    python test_RFB.py -d COCO -v relu_mid_mutil_rate -s 512 -r "9,5,2,1,6,3,2,1,9,5,2,1" -t True -m  $weight --save_folder=$save_folder
done
#python eval_aspp.py --trained_model="./weights/voc_aspp_1259/VOC.pth" --save_folder="./eval/voc_aspp-1259/"
