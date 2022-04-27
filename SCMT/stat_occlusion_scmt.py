# vim: expandtab:ts=4:sw=4
"""
@Filename: stat_occlusion_scmt.py
@Discription: Calculate occlusion score of the SCMT results.
"""

import os
import sys
import numpy as np
import pickle


def read_from_file(src_path):
    """
    Read tracking result file
    """
    print ('* reading file from {}...'.format(src_path))
    c_dict = {}
    for cam in ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']:
        cam_id = int(cam[2:])
        c_dict[cam_id] = {}
        with open(os.path.join(src_path, '{}.txt'.format(cam)), 'r') as fid:
            for line in fid.readlines():
                s = [int(x) for x in line.rstrip().split(',')]
                fr_id, track_id, x, y, w, h = s[:6]
                # if drop_edge_box(cam_id, x, y, w, h, EDGE_THRE): # drop those frames whose boxes are out of the image
                #     continue
                if fr_id not in c_dict[cam_id]:
                    c_dict[cam_id][fr_id] = {}
                    c_dict[cam_id][fr_id]['track_id'] = []
                    c_dict[cam_id][fr_id]['bbox'] = []
                c_dict[cam_id][fr_id]['track_id'].append(track_id)
                c_dict[cam_id][fr_id]['bbox'].append([x, y, w, h])
    return c_dict


def calc_occlusion_score(bboxes, fr_id, cam_id, track_ids):
    """
    Calculate occlusion score
    """
    res_dict = {}
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2] + x1
    y2 = bboxes[:, 3] + y1
    scores = y2
    sorted_ids = scores.argsort()[::-1]
    areas = (x2 - x1) * (y2 - y1)
    keep_ids = []
    # keep_pairs = []
    filtered_ids = set()
    submit_ids = []
    res_dict = {}

    for j in range(len(sorted_ids)):
        i = sorted_ids[j]
        xx1 = np.maximum(x1[i], x1[sorted_ids[j + 1:]])
        yy1 = np.maximum(y1[i], y1[sorted_ids[j + 1:]])
        xx2 = np.minimum(x2[i], x2[sorted_ids[j + 1:]])
        yy2 = np.minimum(y2[i], y2[sorted_ids[j + 1:]])
        
        inter_areas = np.maximum(0., xx2 - xx1 + 1) * np.maximum(0., yy2 - yy1 + 1)
        ious = inter_areas / areas[sorted_ids[j + 1:]]
        max_iou = np.max(ious) if ious.size != 0 else 0

        key = 'c0{}_{}_{}'.format(cam_id, fr_id, track_ids[i])
        res_dict[key] = max_iou

    return res_dict

if __name__ == '__main__':
    src_path = sys.argv[1]
    dst_root = src_path
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    c_dict = read_from_file(src_path)

    print ('* computing occlusion score...')
    for cam_id in c_dict.keys():
        res_dict = {}
        for fr_id in c_dict[cam_id].keys():
            track_ids = np.array(c_dict[cam_id][fr_id]['track_id']) # dim: N
            bboxes = np.array(c_dict[cam_id][fr_id]['bbox']) # dim: N x 4
            res_dict.update(calc_occlusion_score(bboxes, fr_id, cam_id, track_ids))

        with open(os.path.join(dst_root, 'c0{}_truncation.pkl'.format(cam_id)), 'wb') as f:
            pickle.dump(res_dict, f, protocol=2)
