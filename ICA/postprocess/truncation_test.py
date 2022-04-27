"""
truncation tricks for test
"""

#*** coding: utf-8 ***#
import os, sys, pdb
import cv2
import pdb
import numpy as np
import shutil

def to_track_dict(lines, cams):
    """
    convert submit format to tracklets dictionary
    """
    tracklets ={}
    for line in lines:
        line_list = line.strip().split()
        cam_id   = int(line_list[0])
        if cam_id not in cams:
            continue
        obj_id   = int(line_list[1])
        frame_id = int(line_list[2])
        x1       = int(line_list[3])
        y1       = int(line_list[4])
        w        = int(line_list[5])
        h        = int(line_list[6])
        if cam_id not in tracklets.keys():
            tracklets[cam_id] = {}
        if obj_id not in tracklets[cam_id].keys():
            tracklets[cam_id][obj_id] = []
        tracklets[cam_id][obj_id].append([frame_id, x1, y1, w, h])
    return tracklets


def find_ref_and_border_frames(tracklets):
    """
    find border frames and nearest to border frames
    """
    wh_ratio_dict = {}
    both_two_border = {}
    
    for cam_id in tracklets.keys():
        img_w = 1280
        img_h = 720 if cam_id == 45 or cam_id == 46 else 960
        if cam_id not in wh_ratio_dict.keys():
            wh_ratio_dict[cam_id] = {}
        if cam_id not in both_two_border.keys():
            both_two_border[cam_id] = []
        
        for obj_id in tracklets[cam_id].keys():
            if obj_id not in wh_ratio_dict[cam_id].keys():
                wh_ratio_dict[cam_id][obj_id] = {}

            tracklet = tracklets[cam_id][obj_id]
            sorted(tracklet, key=lambda x:x[0])

            previous_status = None ### not in border
            for obj in tracklet:
                frame_id = obj[0]
                x1 = obj[1]
                y1 = obj[2]
                w = obj[3]
                h = obj[4]

                ## only one border
                ## for both two border
                if (x1 < 2 and y1 < 2) or (x1 < 2 and y1 + h > img_h - 2) or \
                        (x1 + w > img_w - 2 and y1 < 2) or (x1 + w > img_w - 2 and y1 + h > img_h - 2):
                            if obj_id not in both_two_border[cam_id]:
                                both_two_border[cam_id].append(obj_id)

                if x1 < 2 or y1 < 2 or x1 + w > img_w - 2 or y1 + h > img_h - 2:
                    if previous_status is not None and previous_status == False:
                        big_ref = frame_id - 1
                        try:
                            assert('big_ref' not in wh_ratio_dict[cam_id][obj_id].keys())
                            wh_ratio_dict[cam_id][obj_id]['big_ref'] = big_ref
                        except:
                            print('big-{}-{}'.format(cam_id, obj_id))
                    previous_status = True
                    if 'border' not in wh_ratio_dict[cam_id][obj_id].keys():
                        wh_ratio_dict[cam_id][obj_id]['border'] = {}
                    assert(frame_id not in wh_ratio_dict[cam_id][obj_id]['border'].keys())
                    wh_ratio_dict[cam_id][obj_id]['border'][frame_id] = [w / h, w * h]
                else:
                    if previous_status is not None and previous_status == True:
                        small_ref = frame_id
                        try:
                            assert('small_ref' not in wh_ratio_dict[cam_id][obj_id].keys())
                            wh_ratio_dict[cam_id][obj_id]['small_ref'] = small_ref
                        except:
                            print('small-{}-{}'.format(cam_id, obj_id))
                    previous_status = False
                    if 'normal' not in wh_ratio_dict[cam_id][obj_id].keys():
                        wh_ratio_dict[cam_id][obj_id]['normal'] = {}

                    # assert(frame_id not in wh_ratio_dict[cam_id][obj_id]['normal'].keys())

                    wh_ratio_dict[cam_id][obj_id]['normal'][frame_id] = [w / h, w * h]
    return wh_ratio_dict, both_two_border


def remove_border_boxes(wh_ratio_dict, both_two_border, only_one_border=True):
    """
    remove truncated boxes
    """
    remove_bboxes = {}
    remove_num = 0
    for cam_id in wh_ratio_dict.keys():
        if cam_id not in remove_bboxes.keys():
            remove_bboxes[cam_id] = {}
        for obj_id in wh_ratio_dict[cam_id].keys():
            if cam_id in both_two_border.keys() and \
               obj_id in both_two_border[cam_id] and \
               only_one_border:
                continue
            if 'border' in wh_ratio_dict[cam_id][obj_id]:
                border_fid = wh_ratio_dict[cam_id][obj_id]['border'].keys()
                if 'small_ref' in wh_ratio_dict[cam_id][obj_id]:
                    small_ref = wh_ratio_dict[cam_id][obj_id]['small_ref']
                else:
                    small_ref = 9999999999999
                if 'big_ref' in wh_ratio_dict[cam_id][obj_id]:
                    big_ref = wh_ratio_dict[cam_id][obj_id]['big_ref']
                else:
                    big_ref = 99999999999999

                name = '{}-{}'.format(cam_id, obj_id)

                mean_f = 3
                big_ref_mean = np.array([0.0, 0.0])
                small_ref_mean = np.array([0.0, 0.0])
                if big_ref < 2200:
                    num = 0
                    for i in range(big_ref - mean_f, big_ref):
                        if i not in wh_ratio_dict[cam_id][obj_id]['normal'].keys():
                            continue
                        num += 1
                        try:
                            big_ref_mean += np.array(wh_ratio_dict[cam_id][obj_id]['normal'][i])
                        except:
                            pdb.set_trace()
                    if num == 0:
                        ref_fm = [key for key in wh_ratio_dict[cam_id][obj_id]['normal'].keys()]
                        ref_fm.sort()
                        big_ref_mean = np.array(wh_ratio_dict[cam_id][obj_id]['normal'][ref_fm[-1]])
                    else:
                        big_ref_mean /= num
                if small_ref < 2200:
                    num = 0
                    for i in range(small_ref, small_ref + mean_f):
                        if i not in wh_ratio_dict[cam_id][obj_id]['normal'].keys(): 
                            continue
                        num += 1
                        small_ref_mean += np.array(wh_ratio_dict[cam_id][obj_id]['normal'][i])
                    if num == 0:
                        ref_fm = [key for key in wh_ratio_dict[cam_id][obj_id]['normal'].keys()]
                        ref_fm.sort()
                        small_ref_mean = np.array(wh_ratio_dict[cam_id][obj_id]['normal'][ref_fm[0]])
                    else:
                        small_ref_mean /= num

                for border_i in border_fid:
                    if abs(border_i - big_ref) < abs(border_i - small_ref) and big_ref < 2200:
                        [wh_ratio, area_ratio] = big_ref_mean / \
                                np.array(wh_ratio_dict[cam_id][obj_id]['border'][border_i])
                        if (wh_ratio > 1.5 or wh_ratio < 0.5) or (area_ratio < 0.7 or area_ratio > 1.3):
                            if obj_id not in remove_bboxes[cam_id].keys():
                                remove_bboxes[cam_id][obj_id] = []
                            remove_bboxes[cam_id][obj_id].append(border_i)
                            remove_num += 1
                    if abs(border_i - big_ref) > abs(border_i - small_ref) and small_ref < 2200:
                        [wh_ratio, area_ratio] = small_ref_mean / \
                                 np.array(wh_ratio_dict[cam_id][obj_id]['border'][border_i])
                        if (wh_ratio > 1.5 or wh_ratio < 0.5) or (area_ratio < 0.7 or area_ratio > 1.3):
                            if obj_id not in remove_bboxes[cam_id].keys():
                                remove_bboxes[cam_id][obj_id] = []
                            remove_bboxes[cam_id][obj_id].append(border_i)
                            remove_num += 1
    return remove_bboxes, remove_num 

def write_output_wo_removed(remove_bboxes, lines):
    """
    write boxes without truncated to submit format
    """
    fw = open(pred_save_file, 'w')
    remove_num = 0
    for line in lines:
        line_list = line.strip().split()
        cam_id   = int(line_list[0])
        obj_id   = int(line_list[1])
        frame_id = int(line_list[2])
        if cam_id in remove_bboxes.keys() and obj_id in remove_bboxes[cam_id].keys():
            if frame_id not in remove_bboxes[cam_id][obj_id]:
                fw.write(line)
            else:
                remove_num += 1
        else:
            fw.write(line)
    print('remove num in write: {}'.format(remove_num))
    fw.close()
    

pred_raw, pred_save_file = sys.argv[1:]

lines = open(pred_raw).readlines()

tracklets = to_track_dict(lines, [41, 42, 43, 44, 45, 46])
wh_ratio_dict, both_two_border = find_ref_and_border_frames(tracklets)

remove_bboxes, remove_num = remove_border_boxes(wh_ratio_dict, both_two_border, only_one_border=False)
print('remove_num: {}'.format(remove_num))

write_output_wo_removed(remove_bboxes, lines)

