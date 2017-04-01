This is my fork of Ross Girshick's [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and has code and models for the WACV 2017 paper [StuffNet: Using 'Stuff' to Improve Object Detection](https://arxiv.org/abs/1610.05861).

Please use [this](https://github.com/samarth-robo/py-faster-rcnn/tree/d26d1e386ec185d22707d06f9f5c8e47a255bc9a) version of the repository. I've created `-seg` versions of the training and solver prototxt files e.g. `experiments/scripts/faster_rcnn_end2end-seg.sh`.

The repository is configured by default for `StuffNet-30` i.e. 30 segmentation classes. To switch to `StuffNet-10`, you will need to:
1. Change the `num_output` parameter in the `train-seg.prototxt` and `test-seg.prototxt` files to 10.
2. Change the `SEG_CLASSES` parameter in `experiments/cfgs/faster_rcnn_end2end-seg.yml` to 10.

# Models:
- [VOC 2007 StuffNet-30](https://drive.google.com/file/d/0B5_6NRwNEqMPNjBXQ2tJZUdaTnM/view?usp=sharing)
- [VOC 2010 StuffNet-10](https://drive.google.com/file/d/0B5_6NRwNEqMPc3ZiUkRhaFZmM00/view?usp=sharing)
- [VOC 2010 StuffNet-30](https://drive.google.com/file/d/0B5_6NRwNEqMPeDE1aGNTR1RUamc/view?usp=sharing)
- [VOC 2012 StuffNet-30](https://drive.google.com/file/d/0B5_6NRwNEqMPUXpPSDdIX0IxTlE/view?usp=sharing)

# Segmentation images:
`StuffNet` models need segmentation images in addition to RGB images with bounding box annotations for training. You should generate them for your dataset using feature constraining (see paper for details) and put them in `DATA_PATH/context_images_SEG_CLASSES/*.ppm`. `SEG_CLASSES` is either 10 or 30. `DATA_PATH` for VOC 2007 is `VOCdevkit/VOC2007`, for VOC 2010 is `VOCdevkit/VOC2010`, and so forth. The names of the PPM files should be exactly the same as the corresponding RGB images. For example, if the RGB image is `DATA_PATH/JPEGImages/2010_006993.jpg` the segmentation image for training `StuffNet-10` should be `DATA_PATH/context_images_10/2010_006993.ppm`.
