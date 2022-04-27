"""
Date: 2022/04/19
"""

import os, sys, pdb
import numpy as np
import math

# for angle mode, it is deprecated in the latest version!
thres = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (-1, -1)]

zone1_in = [(458, 1280, -1, 231), (500, 1010, -1, 180), (500, 1280, -1, 220), 
            (-1, 430, 150, 500), (-1, 500, -1, 200), (700, 1240, -1, 170)]
zone2_out = [(788, 1280, -1, 311), (510, 1120, 90, 300), (700, 1280, 150, 400), 
             (280, 730, 110, 250), (350, 730, 50, 220), (900, 1280, 140, 300)]
zone3_in = [(-1, 602, 366, 960), (-1, 500, 440, 960), (-1, 530, 370, 960), 
            (830, 1280, 235, 570), (1060, 1280, 185, 500), (-1, 400, 350, 720)]
zone4_out = [(-1, 522, 167, 361), (-1, 450, 200, 420), (-1, 430, 160, 370),
             (470, 1280, 500, 960), (50, 1280, 400, 720), (-1, 450, 190, 345)]
zone5 = [(-1, 388, -1, 170), (-1, 420, -1, 210), (-1, 420, -1, 200), 
         (-1, 10, 900, 960), (-1, 310, 210, 540), (-1, 610, -1, 200)]
zone6 = [(782, 1280, 311, 960), (710, 1280, 340, 960), (700, 1280, 400, 960), 
         (719, 1280, -1, 220), (732, 1280, -1, 170), (510, 1280, 305, 720)]

class Tracklet(object):
    """ This class is used to record all necessary information and calculate the trajectory of this tracklet.
    The tracklet keeps all information of each tracker. The main usage of this object is to find the "in port" and "out port" of this tracklet.
    
    Attributes:
        cam_id: camera id
        x, y: the central coordinates of this track in the special frame
        fr_id: frame id
        theta: the angle of this track in the special frame
        ag_id: the angle key in the pre-defined dict of movement direction
    """

    def __init__(self, cam_id, x, y, fr_id, theta, ag_id):
        """ Initialize TrackletClass.
        """
        self.cam_id = cam_id
        self.co = [[x, y]]
        self.frames = [fr_id]
        self.thetas = [theta]
        self.ag_ids = [ag_id]
        self.valid_ag_ids = []
        self.st_id = -1 # the "in port" id when the track appears in the image at the start
        self.en_id = -1 # the "out port" id when the track disappears in the image at the end
        self.select_st_id(x, y)

    def select_st_id(self, x, y):
        """ For zone mode, it is a simple and efficient method.
        Update the start index with pre-defined zones.
        """
        idx = self.cam_id - 41
        if zone1_in[idx][0] < x < zone1_in[idx][1] and zone1_in[idx][2] < y < zone1_in[idx][3]:
            self.st_id = 1
        elif zone3_in[idx][0] < x < zone3_in[idx][1] and zone3_in[idx][2] < y < zone3_in[idx][3]:
            self.st_id = 3
        elif zone5[idx][0] < x < zone5[idx][1] and zone5[idx][2] < y < zone5[idx][3]:
            self.st_id = 0
        elif zone6[idx][0] < x < zone6[idx][1] and zone6[idx][2] < y < zone6[idx][3]:
            self.st_id = 2
        else:
            # print (x, y)
            # raise AssertionError('error with start id!')
            pass # maybe in the center of this image at the beginning

    def select_en_id(self, x, y):
        """ For zone mode, it is a simple and efficient method.
        Update the end index with pre-defined zones.
        """
        idx = self.cam_id - 41
        if zone2_out[idx][0] < x < zone2_out[idx][1] and zone2_out[idx][2] < y < zone2_out[idx][3]:
            self.en_id = 3
        elif zone4_out[idx][0] < x < zone4_out[idx][1] and zone4_out[idx][2] < y < zone4_out[idx][3]:
            self.en_id = 1
        elif zone5[idx][0] < x < zone5[idx][1] and zone5[idx][2] < y < zone5[idx][3]:
            self.en_id = 2
        elif zone6[idx][0] < x < zone6[idx][1] and zone6[idx][2] < y < zone6[idx][3]:
            self.en_id = 0
        else:
            pass # no update

        # for special case in camera 42 and 43: some tracks would choose wrong out port in the last time.
        if self.st_id == 1 and self.frames[-1] > 1995 and (self.cam_id == 42 or self.cam_id == 43):
            self.en_id = -1
        if self.st_id == 3 and self.frames[-1] > 1995 and (self.cam_id == 45 or self.cam_id == 43):
            self.en_id = -1

        if len(self.frames) < 4:
            self.en_id = -1
        
    def select_st_en_id_with_angle(self):
        """ For angle mode, it is deprecated in the latest version!
        Update the start and end iddex using angles
        """
        if len(self.valid_ag_ids) >= 2:
            n = len(self.valid_ag_ids)
            st_id_list = self.valid_ag_ids[:n // 2]
            en_id_list = self.valid_ag_ids[n // 2:]
            self.st_id = max(st_id_list, key=st_id_list.count)
            self.en_id = max(en_id_list, key=en_id_list.count)

    def cal_angle(self, x, y):
        """ For angle mode, it is deprecated in the latest version!
        Calculate the current movement angle of this track
        """
        x1, y1 = self.co[-1]
        x2, y2 = x, y
        dx, dy = x2 - x1, y2 - y1
        stop_thre_x = 5
        stop_thre_y = 3
        if abs(dx) < stop_thre_x or abs(dy) < stop_thre_y: # the track is stopped
            theta = -1
        elif x2 == x1:
            if y2 == y1:
                theta = 0.
            elif y2 > y1:
                theta = math.pi / 2
            else:
                theta = math.pi * 3 / 2
        elif x2 > x1 and y2 < y1:
            theta = math.pi * 2 + math.atan(dy / dx)
        elif x2 < x1 and y2 <= y1:
            theta = math.pi + math.atan(dy / dx)
        elif x2 < x1 and y2 > y1:
            theta = math.pi + math.atan(dy / dx)
        elif x2 > x1 and y2 >= y1:
            theta = math.atan(dy / dx)
        else:
            print (x1, y1, x2, y2)
            raise AssertionError('error while calculating angel!')

        theta /= math.pi
        return theta

    def select_angle_id(self, theta):
        """ For angle mode, it is deprecated in the latest version!
        Select the current movement angle index of this track
        """
        th1 = thres[0] # right down
        th2 = thres[1] # left down
        th3 = thres[2] # left up
        th4 = thres[3] # right up
        if theta > th1[0] and theta < th1[1]:
            ag_id = 0
        elif theta >= th2[0] and theta < th2[1]:
            ag_id = 1
        elif theta >= th3[0] and theta < th3[1]:
            ag_id = 2
        elif theta >= th4[0] and theta < th4[1]:
            ag_id = 3
        # elif theta == 0: # maybe stopped
            # ag_id = angle_dict[car_id][-1][-1] # just keep last angle id
        else:
            ag_id = 4
            # raise AssertionError('error in no angel id!')
        return ag_id

    def add_element(self, x, y, fr_id):
        """ Key function. 
        It adds information and calculate the trajectory.
        """
        self.co.append((x, y))
        self.frames.append(fr_id)

        theta = self.cal_angle(x, y)
        ag_id = self.select_angle_id(theta)
        self.thetas.append(theta)
        self.ag_ids.append(ag_id)
        if ag_id in [0, 1, 2, 3]:
            self.valid_ag_ids.append(ag_id)
        # self.select_st_en_id_with_angle()

        self.select_en_id(x, y)


if __name__ == '__main__':
    src_root = './tmp/track_results/'
    dst_root = './data/preprocessed_data/'

    cam_list = os.listdir(src_root)
    cam_list.sort()
    for cam_file in cam_list:
        print ('processing: {}'.format(cam_file))
        cam_id = int(cam_file[1:4]) # c04x.txt -> 4x
        tracklet_dict = {}
        with open(os.path.join(src_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(x) for x in line.rstrip().split()] # [cam_id, track_id, frame_id, x, y, w, h, -1, -1]
                c_id, track_id, fr_id, x, y, w, h = s[:-2]
                assert (c_id == cam_id)
                xc = (x + w / 2.)
                yc = (y + h / 2.)
                if s[1] not in tracklet_dict:
                    tracklet_dict[track_id] = Tracklet(c_id, xc, yc, fr_id, -1, 4) # initialized theta=-1 and angle id=4
                else:
                    tracklet_dict[track_id].add_element(xc, yc, fr_id)
