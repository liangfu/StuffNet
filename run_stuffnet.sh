#!/usr/bin/env bash

# ./experiments/scripts/faster_rcnn_end2end-seg.sh 0 vgg16 pascal_voc --vis
set -v
export PYTHONPATH=/home/liangfu/local/python:$PYTHONPATH
export GLOG_minloglevel=0

# python -u ./tools/test_net.py --gpu 0 --def models/pascal_voc/vgg16/faster_rcnn_end2end/test-seg.prototxt --net output/faster_rcnn_end2end/voc_2010_train/vgg16_faster_rcnn-seg_iter_70000.caffemodel --imdb voc_2010_val --cfg experiments/cfgs/faster_rcnn_end2end-seg.yml --comp --vis

python -u ./tools/test_net.py --gpu 0 --def models/cityscapes/vgg16/faster_rcnn_end2end/test-seg.prototxt --net output/faster_rcnn_end2end/cityscapes_2017_train/vgg16_faster_rcnn-seg_iter_300.caffemodel --imdb cityscapes_val --cfg experiments/cfgs/faster_rcnn_end2end-seg.yml --comp --vis

# python -u ./tools/train_net.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end-seg.yml --solver models/cityscapes/vgg16/faster_rcnn_end2end/solver-seg.prototxt --weights models/imagenet/VGG_ILSVRC_16_layers.caffemodel --imdb cityscapes_train

set +v

