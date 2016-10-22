import numpy as np
import os.path as osp
import sys
import caffe
import cv2
import _init_paths
import matplotlib.pyplot as plt
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from IPython.core.debugger import Tracer

caffe.set_mode_gpu()
caffe.set_device(0)

class SegLabelsGenerator:
  def __init__(self, dataset, split, year):

    # object accessing dataset
    if dataset == 'coco':
      self.D = coco(split, year)
    elif dataset == 'pascal_voc':
      self.D = pascal_voc(split, year)
    else:
      print 'Wrong dataset {:s}'.format(dataset)
      sys.exit(-1)
    
    self.mean_pixel = np.array([102.9801, 115.9465, 122.7717])  # BGR

    # net
    self.net = caffe.Net('../models/seg/deploy.prototxt',
        '../output/faster_rcnn_end2end/voc_2010_trainval/stuffnet_simple_30.caffemodel',
        caffe.TEST)

  def get_seg(self, im):
    im_in = np.zeros((1, im.shape[2], im.shape[0], im.shape[1]),
        dtype=np.float32)
    im_in[0, :, :, :] = (im.astype(np.float32) -
        self.mean_pixel).transpose((2, 0, 1))
    self.net.blobs[self.net.inputs[0]].reshape(im_in.shape[0], im_in.shape[1],
        im_in.shape[2], im_in.shape[3])
    self.net.reshape()
    self.net.blobs[self.net.inputs[0]].data[...] = im_in
    probs = self.net.forward()
    probs = probs['prob_seg']
    probs = np.squeeze(probs).transpose((1, 2, 0))
    labels = np.argmax(probs, axis=2).astype(np.uint8)
    return labels

  def save_segs(self):
    for count, im_idx in enumerate(self.D._image_index):
      if count % 100 == 0:
        print 'Image {:d} of {:d}'.format(count, len(self.D._image_index))
      im_filename = self.D.image_path_from_index(im_idx)
      seg_filename = self.D.seg_path_from_index(im_idx)
      im = cv2.imread(im_filename)
      if im is None:
        print 'Could not read ', im_filename
        sys.exit(-1)
      seg = self.get_seg(im);
      cv2.imwrite(seg_filename, seg)

  def coco_modify_segs(self):
    for count, im_idx in enumerate(self.D._image_index):
      if count % 100 == 0:
        print 'Image {:d} of {:d}'.format(count, len(self.D._image_index))
      seg_filename = self.D.seg_path_from_index(im_idx)
      seg = cv2.imread(seg_filename, -1)
      if seg is None:
        print 'Could not read', seg_filename

      # get annotations
      c = self.D._COCO
      ann_ids = c.getAnnIds(imgIds=im_idx, iscrowd=False)
      anns = c.loadAnns(ann_ids)
      mask = np.zeros(seg.shape, dtype=np.int)
      for ann in anns:
        if 'segmentation' not in ann:
          continue
        # TODO verify 9
        cat_id = ann['category_id'] + 9
        if type(ann['segmentation']) == list:
          for s in ann['segmentation']:
            poly = np.array(s).reshape((len(seg)/2, 2))
            poly = Polygon(poly)
            pth = path.Path(poly.get_xy(), closed=True)
            y, x = np.mgrid[:seg.shape[0], :seg.shape[1]]
            points = np.transpose((x.ravel(), y.ravel()))
            m = pth.contains_points(points)
            mask += (cat_id * m)
        else:
          Tracer()()


if __name__ == '__main__':
  if len(sys.argv) != 4:
    print 'Usage: python {:s} dataset split year'.format(sys.argv[0])
    sys.exit(-1)

  lg = SegLabelsGenerator(sys.argv[1], sys.argv[2], sys.argv[3])
  lg.save_segs()

  # if len(sys.argv) != 2:
  #   print 'Usage: python {:s} image'.format(sys.argv[0])
  #   sys.exit(-1)

  # lg = SegLabelsGenerator('2010')
  # im = cv2.imread(sys.argv[1])
  # if im is None:
  #   print 'Could not read ', sys.argv[1]
  #   sys.exit(-1)
  # labels = lg.get_seg(im)
  # plt.imshow(labels)
  # plt.show()
