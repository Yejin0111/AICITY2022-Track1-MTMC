track_version=$1

# store files in data with softlink
if [ ! -d data/track_feats/$track_version ]; then
    mkdir -p data/track_feats/$track_version
fi
ls data/$track_version | grep c04..pkl | xargs -I {} ln -s $PWD/data/$track_version/{} data/track_feats/$track_version

if [ ! -d data/track_results/$track_version ]; then
    mkdir -p data/track_results/$track_version
fi
ln -s $PWD/data/$track_version/*.txt data/track_results/$track_version

if [ ! -d data/truncation_rates/$track_version ]; then
    mkdir -p data/truncation_rates/$track_version
fi
ln -s $PWD/data/$track_version/*_truncation.pkl data/truncation_rates/$track_version


# preprocess the original data
python preprocess.py --src_root data/track_results/$track_version --dst_root data/preprocessed_data/$track_version --feat_root data/track_feats/$track_version --trun_root data/truncation_rates/$track_version

# create the tracking results of single camera.
cat tmp/*.txt > data/preprocessed_data/$track_version/all_cameras.txt

# multi-camera matching and generate final results
python multi_camera_matching.py --src_root data/preprocessed_data/$track_version --dst_root submit/$track_version --st_dim 0 --en_dim 2048

# postprocess
# occlusion
python postprocess/filter_occlusion.py submit/"$track_version"/track1.txt submit/"$track_version"/postprocess
# expand box
python postprocess/expand_pred_boxes.py submit/"$track_version"/postprocess/submit_result.txt submit/"$track_version"/postprocess/submit_result_expand_1.3.txt
# truncation
python postprocess/truncation_test.py submit/"$track_version"/postprocess/submit_result_expand_1.3.txt submit/"$track_version"/postprocess/submit_result_expand_1.3_truncation.txt
# drop small targets from roi mask
python postprocess/filter_roi_result.py submit/"$track_version"/postprocess/submit_result_expand_1.3_truncation.txt submit/"$track_version"/postprocess/submit_result_expand_1.3_truncation_filter_roi_result.txt
