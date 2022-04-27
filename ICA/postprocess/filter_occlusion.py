import os, sys, pdb
import numpy as np
import json
from tqdm import tqdm


IOU_THRE = 0.8 
IOU_THRE_TL = 0.3
IOU_THRE_T = 0.3

def read_from_file(src_path):
    print ('* reading file from {}...'.format(src_path))
    c_dict = {}
    with open(src_path, 'r') as fid:
        for line in fid.readlines():
            s = [int(x) for x in line.rstrip().split()]
            cam_id, track_id, fr_id, x, y, w, h = s[:7]
            # if drop_edge_box(cam_id, x, y, w, h, EDGE_THRE): # drop those frames whose boxes are out of the image
            #     continue

            if cam_id not in c_dict:
                c_dict[cam_id] = {}
            if fr_id not in c_dict[cam_id]:
                c_dict[cam_id][fr_id] = {}
                c_dict[cam_id][fr_id]['track_id'] = []
                c_dict[cam_id][fr_id]['bbox'] = []
            c_dict[cam_id][fr_id]['track_id'].append(track_id)
            c_dict[cam_id][fr_id]['bbox'].append([x, y, w, h])
    return c_dict

def write_output(c_dict, dst_root):
    keep_obj = open(os.path.join(dst_root, 'keep_result.txt'), 'w')
    filter_obj = open(os.path.join(dst_root, 'filter_result.txt'), 'w')
    submit_obj = open(os.path.join(dst_root, 'submit_result.txt'), 'w')
    cam_id_list = list(c_dict.keys())
    cam_id_list.sort()
    for cam_id in cam_id_list:
        fr_id_list = list(c_dict[cam_id].keys())
        fr_id_list.sort()
        for fr_id in fr_id_list:
            track_ids = c_dict[cam_id][fr_id]['track_id']
            bboxes = c_dict[cam_id][fr_id]['bbox']
            for i in range(track_ids.size):
                keep_obj.write('{} {} {} {} -1 -1\n'.format(cam_id, track_ids[i], fr_id, ' '.join(map(str, bboxes[i]))))
            filter_track_ids = c_dict[cam_id][fr_id]['filter_track_id']
            filter_bboxes = c_dict[cam_id][fr_id]['filter_bbox']
            for i in range(filter_track_ids.size):
                filter_obj.write('{} {} {} {} -1 -1\n'.format(cam_id, filter_track_ids[i], fr_id, ' '.join(map(str, filter_bboxes[i]))))
            submit_track_ids = c_dict[cam_id][fr_id]['submit_track_id']
            submit_bboxes = c_dict[cam_id][fr_id]['submit_bbox']
            for i in range(submit_track_ids.size):
                submit_obj.write('{} {} {} {} -1 -1\n'.format(cam_id, submit_track_ids[i], fr_id, ' '.join(map(str, submit_bboxes[i]))))
    keep_obj.close()
    filter_obj.close()

def nms(bboxes, fr_id, track_ids):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2] + x1
    y2 = bboxes[:, 3] + y1
    scores = y2
    sorted_ids = scores.argsort()[::-1]
    areas = (x2 - x1) * (y2 - y1)
    keep_ids = []
    filtered_ids = set()
    submit_ids = []

    for j in range(len(sorted_ids)):
        i = sorted_ids[j]

        if j == (len(sorted_ids) - 1):
            if i not in filtered_ids:
                keep_ids.append(i)
                submit_ids.append(i)
            continue
        
        xx1 = np.maximum(x1[i], x1[sorted_ids[j+1:]])
        yy1 = np.maximum(y1[i], y1[sorted_ids[j+1:]])
        xx2 = np.minimum(x2[i], x2[sorted_ids[j+1:]])
        yy2 = np.minimum(y2[i], y2[sorted_ids[j+1:]])
        
        inter_areas = np.maximum(0., xx2 - xx1 + 1) * np.maximum(0., yy2 - yy1 + 1)
        ious = inter_areas / areas[sorted_ids[j+1:]]
        max_iou = np.max(ious)
        xc, yc = (x1[i] + x2[i]) / 2, (y1[i] + y2[i]) / 2
        if xc > 458 and yc < 180: # c041, top
            thre = IOU_THRE_T
        elif xc < 458 and yc < 230: # c041, top left
            thre = IOU_THRE_TL
        else:
            thre = IOU_THRE

        if max_iou > thre:
            k = sorted_ids[np.argmax(ious) + j + 1]
            if i not in filtered_ids:
                keep_ids.append(i)
                submit_ids.append(i)
            filtered_ids.add(k)
        else:
            if i not in filtered_ids:
                submit_ids.append(i)
    keep_ids = list(set(keep_ids))
    filtered_ids = list(filtered_ids)
    assert len(set(submit_ids)) == len(submit_ids)
    return keep_ids, filtered_ids, submit_ids

if __name__ == '__main__':
    src_path, dst_root = sys.argv[1:]

    c_dict = read_from_file(src_path)

    print ('* computing nms...')
    for cam_id in c_dict.keys():
        for fr_id in c_dict[cam_id].keys():
            track_ids = np.array(c_dict[cam_id][fr_id]['track_id']) # dim: N
            bboxes = np.array(c_dict[cam_id][fr_id]['bbox']) # dim: N x 4

            keep_ids, filtered_ids, submit_ids = nms(bboxes, fr_id, track_ids)
            c_dict[cam_id][fr_id]['track_id'] = track_ids[keep_ids]
            c_dict[cam_id][fr_id]['bbox'] = bboxes[keep_ids]
            c_dict[cam_id][fr_id]['filter_track_id'] = track_ids[filtered_ids]
            c_dict[cam_id][fr_id]['filter_bbox'] = bboxes[filtered_ids]
            c_dict[cam_id][fr_id]['submit_track_id'] = track_ids[submit_ids]
            c_dict[cam_id][fr_id]['submit_bbox'] = bboxes[submit_ids]

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    print ('* writing file to {}...'.format(dst_root))
    write_output(c_dict, dst_root)
