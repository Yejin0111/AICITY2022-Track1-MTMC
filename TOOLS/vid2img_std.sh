#!/usr/bin/env sh

video_path=$1
image_path=$2
[ ! -d $image_path ] && mkdir -p $image_path
#ffmpeg -i $video_path -vf fps=fps=3 $image_path/$4_%05d.jpg
ffmpeg -i $video_path -r 6 -q:v 1 $image_path/$4_%05d.jpg
#ffmpeg -i $video_path -vframes 1 -q:v 1 $image_path/$4_%05d.jpg
