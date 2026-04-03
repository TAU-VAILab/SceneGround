#!/bin/bash

set -ex
cathedral_number=$1
colmap_number=$2
feature_type=$3
near_plane=$4
loss_percentile=$5
angle=$6
translation=$7
extra=$8
image_percent=$9
initialization=${10}
ignore_gt=1

nerfstudio_cathderals=./local/nerfstudio_data/cathedrals/

# Base Model
nerfstudio_base_path=$nerfstudio_cathderals/$cathedral_number/base_google_earth
# GT
nerfstudio_gt_path=$nerfstudio_cathderals/$cathedral_number/gt
# Meta Image
meta_image_sparse_path=./local/colmap_models/$meta_image_cathedral_number/meta_image/$colmap_number/sparse
# Wild on Base Registration
nerfstudio_wild_cathedral_dir=$nerfstudio_cathderals/$cathedral_number/wild_registration/$colmap_number

echo "Started full pipeline for cathedral $cathedral_number $colmap_number"


if [ ! -d $nerfstudio_base_path ]; then
    echo "Creating Base Model"
    ./scripts/colmap/create_base_model.sh $cathedral_number
fi

if [ ! -d $meta_image_sparse_path ]; then
    echo "Creating Meta Image"
    ./scripts/colmap/create_meta_image.sh $cathedral_number $colmap_number $ignore_gt
fi

if [ ! -d $nerfstudio_wild_cathedral_dir ]; then
    echo "Creating Wild on Base Registration"
    ./scripts/colmap/create_wild_on_base_registration.sh $cathedral_number $colmap_number $ignore_gt
fi

echo "Creating Features"
./scripts/pipelines/features_pipeline.sh $cathedral_number $feature_type $colmap_number $near_plane $loss_percentile $angle $translation $extra $image_percent $initialization

echo "Finished full pipeline for cathedral $cathedral_number $colmap_number"