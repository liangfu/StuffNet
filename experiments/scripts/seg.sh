#!/bin/bash
# Usage:
# ./experiments/scripts/seg.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/seg.sh 0 VGG_CNN_M_1024 pascal_voc \
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
    ITERS=30000
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
LOG="experiments/logs/seg_${NET}_${EXTRA_ARGS_SLUG}.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu ${GPU_ID} \
#   --solver models/seg/solver.prototxt \
#   --weights output/faster_rcnn_end2end/voc_2010_train/vgg16_faster_rcnn_iter_40000.caffemodel \
#   --imdb ${TRAIN_IMDB} \
#   --iters ${ITERS} \
#   --cfg experiments/cfgs/seg.yml \
#   ${EXTRA_ARGS}

set +x
NET_FINAL=output/seg/voc_2010_train/vgg16_seg_iter_30000.caffemodel
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/seg/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/seg.yml \
  --comp \
  ${EXTRA_ARGS}
