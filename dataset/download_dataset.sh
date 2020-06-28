#!/usr/bin/bash

function download_link {
    file=$1
    link=$2
    if [[ ! -f $file ]]; then
        echo "Downloading $file from $link"
        wget -O "$file" "$link"
    else
        echo "$file exists!"
    fi
}

function download_kitti_sceneflow {
    mkdir -p kitti_scene_flow
    pushd kitti_scene_flow

    echo "Scene Flow Evaluation 2015"
    download_link 'data_scene_flow.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip'
    download_link 'data_scene_flow_calib.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip'
    download_link 'data_scene_flow_multiview.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_multiview.zip'
    download_link 'devkit_scene_flow.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_scene_flow.zip'

    popd
}

function download_kitti_flow_2012 {
    mkdir -p kitti_stereo_flow
    pushd kitti_stereo_flow

    echo "Optical Flow Evaluation 2012"

    download_link 'data_stereo_flow.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip'
    download_link 'data_stereo_flow_calib.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow_calib.zip'
    download_link 'data_stereo_flow_multiview.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow_multiview.zip'
    download_link 'devkit_stereo_flow.zip' 'https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_stereo_flow.zip'

    popd
}

function download_flying_chairs {
    mkdir -p flying_chairs
    pushd flying_chairs

    echo "Flying chairs"

    download_link 'FlyingChairs.zip' 'https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip'
    download_link 'FlyingChairs_train_val.txt' 'https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt'
    download_link 'FlyingChairs2.zip' 'https://lmb.informatik.uni-freiburg.de/data/FlyingChairs2.zip'

    popd
}

download_kitti_sceneflow
download_kitti_flow_2012
download_flying_chairs

