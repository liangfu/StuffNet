#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2010_train"
    TEST_IMDB="voc_2010_val"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/faster_rcnn_end2end-seg_${NET}_${EXTRA_ARGS_SLUG}.txt"
exec &> >(tee "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver-seg.prototxt \
  --weights data/imagenet_models/seg_vgg_init.caffemodel,data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end-seg.yml \
  ${EXTRA_ARGS}

set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
NET_FINAL=output/faster_rcnn_end2end/voc_2010_train/vgg16_faster_rcnn-seg_iter_70000.caffemodel
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test-seg.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end-seg.yml \
  --comp \
  ${EXTRA_ARGS}
