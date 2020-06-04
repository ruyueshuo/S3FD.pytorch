#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os.path as osp

import cv2
import time
import numpy as np
from PIL import Image
import scipy.io as sio

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='s3fd evaluatuon wider')
parser.add_argument('--model', type=str,
                    default='/home/ubuntu/code/S3FD.pytorch-master/weights/sfd_face_120000.pth', help='trained model')
parser.add_argument('--thresh', default=0.05, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect_face(net, img, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))

    if use_cuda:
        x = x.cuda()
    # print(x.size())
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    detections = y.data
    detections = detections.cpu().numpy()

    det_conf = detections[0, 1, :, 0]
    det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
    det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
    det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
    det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= args.thresh)[0]
    det = det[keep_index, :]

    return det, t2-t1


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f, _ = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s, _ = detect_face(net, image, st)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b, _ = detect_face(net, image, bt)

    # enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)[0]))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)[0]))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


def get_data():
    subset = 'val'
    if subset is 'val':
        wider_face = sio.loadmat(
            '/home/ubuntu/code/fengda/S3FD.pytorch/eval_tools/wider_face_val.mat')
    else:
        wider_face = sio.loadmat(
            '/home/ubuntu/code/fengda/S3FD.pytorch/eval_tools/wider_face_test.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    imgs_path = os.path.join(
        cfg.FACE.WIDER_DIR, 'WIDER_{}'.format(subset), 'images')
    # save_path = 'eval_tools/s3fd_{}'.format(subset)
    save_path = '/home/ubuntu/sda/test_results/s3fd_{}'.format(subset)

    return event_list, file_list, imgs_path, save_path


def rescale_image(img, scale):
    img = cv2.resize(img, None, None,fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img


def low_res_test(event_list, file_list, imgs_path, save_path, shrink):

    counter = 0
    time_sum = 0
    n = 0
    for index, event in enumerate(event_list):
        # if event[0][0] != "19--Couple":
        #     continue
        filelist = file_list[index][0]
        print("save_path:{}".format(save_path))
        print(event[0][0])
        path = os.path.join(save_path, event[0][0])  # .encode('utf-8')
        if not os.path.exists(path):
            os.makedirs(path)

        for num, file in enumerate(filelist):
            im_name = file[0][0]  # .encode('utf-8')
            in_file = os.path.join(imgs_path, event[0][0], im_name[:] + '.jpg')
            #img = cv2.imread(in_file)
            img = Image.open(in_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)

            max_im_shrink = np.sqrt(
                1700 * 1200 / (img.shape[0] * img.shape[1]))

            # shrink = max_im_shrink if max_im_shrink < 1 else 1
            counter += 1

            try:
                dets, delta_t = detect_face(net, img, shrink)
                time_sum += delta_t
                n += 1

            except RuntimeError as e:
                print(im_name, e)
                dets = np.zeros((0, 1))

            fout = open(osp.join(save_path, event[0][
                0], im_name + '.txt'), 'w')  # .encode('utf-8')
            fout.write('{:s}\n'.format(event[0][0] + '/' + im_name + '.jpg'))  # .encode('utf-8')
            fout.write('{:d}\n'.format(dets.shape[0]))
            for i in range(dets.shape[0]):
                xmin = dets[i][0]
                ymin = dets[i][1]
                xmax = dets[i][2]
                ymax = dets[i][3]
                score = dets[i][4]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                            format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))

    print("average time cost per frame is: %.4f" % (time_sum/n))

    return time_sum/n


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    event_list, file_list, imgs_path, save_dir = get_data()
    cfg.USE_NMS = False
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)
    counter = 0
    time_sum = 0
    n = 0
    scales = [1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9]
    # avg time cost:[0.1319, 0.0402, 0.0230, 0.0167, 0.0135, 0.0140, 0.0125, 0.0096] s/frame
    # fps: [7.58, 24.88, 43.48, 59.88, 74.07, 71.43, 80, 104]
    time_list = {}
    des = save_dir.split('/')
    save_base = os.path.dirname(save_dir)
    save_folder = os.path.basename(save_dir)

    # test over different scale factors.
    for scale in scales:
        # if scale not in [1/8]:
        #     continue
        print("scale factor:{}".format(scale))
        save_path = os.path.join(save_base, (save_folder + '_' + str(scale)))
        time_avg = low_res_test(event_list, file_list, imgs_path, save_path, scale)
        time_list[scale] = time_avg
