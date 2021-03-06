# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from roi_data_layer.roidb import get_seg_path
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
from IPython.core.debugger import Tracer

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    if cfg.TEST.SEG:
        seg_scores = blobs_out['prob_seg']
        seg_scores = np.squeeze(seg_scores).transpose((1, 2, 0))
        return scores, pred_boxes, seg_scores
    else:
        return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.7):
    """Visual debugging of detections."""
    import cv2
    import random
    color_white = (255, 255, 255)
    disp = im.copy()
    flag = False
    for det in dets:
        bbox = det[:4] * 1.
        score = det[-1]
        bbox = map(int, bbox)
        print(bbox)
        if score > thresh:
            cv2.rectangle(disp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_white, thickness=2)
            flag = True
            text = '%s %.3f' % (class_name, score)
            fontFace = cv2.FONT_HERSHEY_PLAIN
            fontScale = 1
            thickness = 1
            textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
            cv2.rectangle(disp, (bbox[0], bbox[1]-textSize[1]), (bbox[0]+textSize[0], bbox[1]), color=(128,0,0), thickness=-1)
            cv2.putText(disp, text, (bbox[0], bbox[1]), color=color_white, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    if flag:
        cv2.imshow("result_"+class_name, disp)
        [exit(0) if cv2.waitKey()&0xff==27 else None]

def vis_segmentation(im, seg_labels):
    disp = (seg_labels*10).astype(np.uint8)
    cv2.imshow("result", disp)
    [exit(0) if cv2.waitKey()&0xff==27 else None]

def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    disp = draw_all_detection(im_array, detections, class_names, scale)
    cv2.imshow("result", disp)
    [exit(0) if cv2.waitKey()&0xff==27 else None]

def draw_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    color_white = (255, 255, 255)
    im = im_array
    color = (0,0,192)
    for j, det in enumerate(detections):
        bbox = det[:4] * scale
        score = det[-1]
        bbox = map(int, bbox)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
        text = '%s %.3f' % (class_names[j], score)
        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        thickness = 1
        textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        cv2.rectangle(im, (bbox[0], bbox[1]-textSize[1]), (bbox[0]+textSize[0], bbox[1]), color=(128,0,0), thickness=-1)
        cv2.putText(im, text, (bbox[0], bbox[1]),
                    color=color_white, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    return im

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if cfg.TEST.SEG:
        n_seg_classes = cfg.SEG_CLASSES
        confcounts = np.zeros((n_seg_classes, n_seg_classes))

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        im = cv2.resize(im, (1024, 512), interpolation=cv2.INTER_LINEAR)
        _t['im_detect'].tic()
        if cfg.TEST.SEG:
            seg_gt = cv2.imread(get_seg_path(imdb._data_path, imdb.image_path_at(i)), -1)
            if seg_gt is None:
                print 'Could not read ', get_seg_path(imdb._data_path, imdb.image_path_at(i))
            scores, boxes, seg_scores = im_detect(net, im, box_proposals)
        else:
            scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :].astype(np.int32)
            # if vis:
            #     vis_detections(im, imdb.classes[j], cls_dets)
            # print cls_dets
            all_boxes[j][i] = cls_dets
            
        if vis:
            classinfo = np.argmax(scores[:,:],axis=1)
            indices = np.where(classinfo>0)[0]
            # indices = np.arange(100)
            detections = []
            class_names = []
            for ind in indices.tolist():
                cls = classinfo[ind]
                box = boxes[ind,cls*4:cls*4+4]
                score = scores[ind,cls]
                detections.append(box.tolist()+[score])
                class_names.append(imdb.classes[cls])
            detections = np.array(detections,np.float32)
            if True: # enable NMS
                indices = nms(detections, cfg.TEST.NMS)
                if len(indices)>0:
                    detections = detections[indices,:]
                    class_names = [class_names[j] for j in indices]
            vis_all_detection(im, detections, class_names, 1.0)

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()
        
        if cfg.TEST.SEG:
          # evaluate the segmentation
          seg_labels = np.argmax(seg_scores, axis=2).astype(int)
          if vis:
              vis_segmentation(im, seg_labels)
          seg_labels = cv2.resize(seg_labels, (seg_gt.shape[1], seg_gt.shape[0]),
              interpolation=cv2.INTER_NEAREST)
          # sumim = seg_gt + seg_labels * n_seg_classes
          # hs = np.bincount(sumim.flatten(), minlength=n_seg_classes*n_seg_classes)
          # confcounts += hs.reshape((n_seg_classes, n_seg_classes))
          # print 'Segmentation evaluation'
          # conf = 100.0 * np.divide(confcounts, 1e-20 + confcounts.sum(axis=1))
          # np.save(output_dir + '/seg_confusion.npy', conf)
          # acc = np.zeros(n_seg_classes)
          # for j in xrange(n_seg_classes):
          #     gtj  = sum(confcounts[j, :])
          #     resj = sum(confcounts[:, j])
          #     gtresj = confcounts[j, j]
          #     acc[j] = 100.0 * gtresj / (gtj + resj - gtresj)
          # print 'Accuracies', acc
          # print 'Mean accuracy', np.mean(acc)
          resname = get_seg_path(imdb._data_path, imdb.image_path_at(i))
          resname = resname.replace("SegmentationClass","results").replace("labelTrainIds","results")
          lut = np.zeros(256)
          lut[:19]=np.array([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33])
          seg_labels = cv2.LUT(seg_labels.astype(np.uint8),lut)
          if cv2.imwrite(resname, seg_labels):
              print("segmentationResult: "+resname)
          else:
              raise Exception("Fail to write segmentation result: "+resname)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
        # if i>10: break

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    # with open(det_file, 'rb') as f:
    #     all_boxes = cPickle.load(f)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
