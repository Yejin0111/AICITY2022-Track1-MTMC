"""
@Filename: run_aicity.py
@Discription: Run AICity Track
"""
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run

if __name__ == '__main__':
    #print(opt)
    for i, seq in enumerate(opt.sequences, start=1):
        print('processing the {}th video {}...'.format(i, seq))
        path_save = join(opt.dir_save, seq + '.txt')
        run(
            sequence_dir=join(opt.dir_dataset, seq),
            detection_file=join(opt.dir_dets, seq + '.npy'),
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=False
        )




