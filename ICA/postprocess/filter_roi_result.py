# vim: expandtab:ts=4:sw=4
"""
@Filename: filter_roi_result.py
@Discription: filter small targets.
"""

import os
import sys
import cv2


if __name__ == '__main__':
    cam_roi = {}
    pair_output = {}
    for cam in ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']:
        cam_roi[cam] = cv2.imread(os.path.join('roi_mask', 'roi_{}.png'.format(cam)))

    base_file, out_file = sys.argv[1:]
    output = ''
    with open(base_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cid, tid, fid, x, y, w, h, _, _ = map(int, line.strip().split())
            cx = round(x + w / 2)
            cy = round(y + h / 2)
            cam = 'c0{}'.format(cid)
            if cam_roi[cam][cy, cx, 0] / 255 == 1:
                output += line
    # out_file = 'final_result.txt'
    with open(out_file, 'w') as f:
        f.write(output)            
