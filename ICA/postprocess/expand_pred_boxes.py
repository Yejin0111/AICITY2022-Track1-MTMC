#*** coding: utf-8  ***#
import os, sys, pdb
import numpy as np
import cv2

file_name, save_file = sys.argv[1:]

mask_rois = 'roi_mask/'
img_shape = {}
for mask_file in os.listdir(mask_rois):
    roi_data = cv2.imread(os.path.join(mask_rois, mask_file))
    h, w, _ = roi_data.shape
    img_shape[mask_file.split('_')[-1].split('.')[0]] = [h, w]    

# file_name = 'track1_allocate_ids_89_2666.txt'

lines = open(file_name).readlines()

# save_file = 'track1_allocate_ids_89_2666_expand_1.2.txt'
f_w = open(save_file, 'w')

for line in lines:
    line_list = line.split()
    cam_id   = int(float(line_list[0]))
    obj_id   = int(float(line_list[1]))
    frame_id = int(float(line_list[2]))
    x1       = int(float(line_list[3]))
    y1       = int(float(line_list[4]))
    w        = int(float(line_list[5]))
    h        = int(float(line_list[6]))
    x2 = x1 + w
    y2 = y1 + h

    #import pdb; pdb.set_trace()
    [height, width] = img_shape['c0' + str(cam_id)]
    
    cx = 0.5*x1 + 0.5*x2
    cy = 0.5*y1 + 0.5*y2

    if w < 120:
        w = 1.2 * w
    else:
        w = 20 + w

    if h < 120:
        h = 1.2 *h
    else:
        h = 20 + h
    #w = min(w*1.2, w+40)
    #h = min(h*1.2, h+40)
    #w = min(w*1.3, w+40)
    #h = min(h*1.3, h+40)
    #w = min(w*1.4, w+45)
    #h = min(h*1.4, h+45)
    x1, y1 = max(0, cx - 0.5*w), max(0, cy - 0.5*h)
    x2, y2 = min(width, cx + 0.5*w), min(height, cy + 0.5*h)
    w , h = x2-x1 , y2-y1

    f_w.write(str(cam_id) + ' ' + str(obj_id) + ' ' + str(frame_id) + ' ' + str(int(x1)) + ' ' + str(int(y1)) + ' ' + str(int(w)) + ' ' + str(int(h)) + ' -1 -1' '\n')
f_w.close()
