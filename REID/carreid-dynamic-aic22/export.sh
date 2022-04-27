#source ./set_cuda10_py3.sh
export CUDA_VISIBLE_DEVICES=3
#python3 tools/export_model.py \
#    -c ./ppcls/configs/truck_resnet50.yaml \
#    -o Global.pretrained_model=weights/latest

#python3 tools/export_model.py \
#    -c ./ppcls/configs/car_resnet50.yaml \
#    -o Global.pretrained_model=weights/car/epoch_49


#python3 tools/export_model.py \
#    -c ./ppcls/configs/car_resnet34_tricks.yaml \
#    -o Global.pretrained_model=weights/r34/epoch_100

#python3 tools/export_model.py \
#    -c ./ppcls/configs/car_swin.yaml \
#    -o Global.pretrained_model=weights/swin/epoch_2

python3 tools/export_model.py \
    -c ./ppcls/configs/car_convNeXt.yaml \
    -o Global.pretrained_model=weights/convnext/epoch_25

#python3 tools/export_model.py \
#    -c ./ppcls/configs/car_poolformer.yaml \
#    -o Global.pretrained_model=weights/poolformer/epoch_100
