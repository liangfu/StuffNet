import numpy as np
import os.path as osp
import sys
import caffe
import cv2
import glob
from IPython.core.debugger import Tracer

caffe.set_mode_gpu()
caffe.set_device(0)

class SegLabelsGenerator:
  def __init__(self, year):
    self.year = year
    
    # paths
    devkit_path = 'data/VOCdevkit2007/VOC{:s}/'.format(self.year)
    self.im_path = osp.join(devkit_path, 'JPEGImages')
    self.seg_path = osp.join(devkit_path, 'context_images_30')
    self.mean_pixel = np.array([102.9801, 115.9465, 122.7717])  # BGR

    # net
    self.net = caffe.Net('models/seg/deploy.prototxt',
        'output/faster_rcnn_end2end/voc_2010_trainval/stuffnet_simple_30.caffemodel',
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
    # get list of images
    im_filenames = glob.glob(self.im_path + '/*.jpg')
    for count, im_filename in enumerate(im_filenames):
      if count % 100 == 0:
        print 'Image {:d} of {:d}'.format(count, len(im_filenames))
      im = cv2.imread(im_filename)
      if im is None:
        print 'Could not read ', im_filename
        sys.exit(-1)
      seg = self.get_seg(im);
      idx = im_filename.split('/')[-1][:-4]
      seg_filename = osp.join(self.seg_path, idx+'.ppm')
      cv2.imwrite(seg_filename, seg)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Usage: python {:s} year'.format(sys.argv[0])
    sys.exit(-1)

  lg = SegLabelsGenerator(sys.argv[1])
  lg.save_segs()
