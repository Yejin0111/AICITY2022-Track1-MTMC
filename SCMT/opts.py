"""
@Filename: opts.py
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    'AICITY': {
        'test':[
            'c041',
            'c042',
            'c043',
            'c044',
            'c045',
            'c046'
        ]
    }
}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'dataset',
            type=str,
            default='AICITY',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            default='test',
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter',
            default=True
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism',
            default=True
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost',
            default=True
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching',
            default=True
        )
        self.parser.add_argument(
            '--root_dataset',
            default='./dataset/CityFlowV2'
        )
        self.parser.add_argument(
            '--dir_save',
            default='./tmp'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.1
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        opt.max_cosine_distance = 0.4
        opt.dir_dets = './dataspace/{}_{}'.format(opt.dataset, opt.mode)
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        opt.sequences = data[opt.dataset][opt.mode]
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            'train' if opt.mode == 'val' else 'test'
        )
        return opt

opt = opts().parse()
